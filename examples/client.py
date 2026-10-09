import argparse
import asyncio
import json
import websockets
import time
try:
    import pyaudio
except ImportError:
    print("Uyarı: 'pyaudio' kütüphanesi bulunamadı. --play argümanı çalışmayacaktır. (Kurulum: pip install pyaudio)")
    pyaudio = None

import threading
import queue
import logging

# Logger kurulumu
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TTS_CLIENT")

class StreamingPlayer:
    def __init__(self, sample_rate=48000, prefetch_count=20, request_sent_time=None):
        self.p = pyaudio.PyAudio()
        self._sample_rate = sample_rate
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=sample_rate,
                                  output=True)
        self.audio_queue = queue.Queue()
        self.prefetch_count = prefetch_count
        self.is_playing = False
        self.player_thread = None
        self._finished = False
        self.request_sent_time = request_sent_time
        self.first_chunk_time = None

    def push_audio(self, chunk_data):
        if self.first_chunk_time is None:
            self.first_chunk_time = time.time()
            
        self.audio_queue.put(chunk_data)
        
        # Sadece başlangıçta belirlenen prefetch sayısına ulaşmayı bekliyoruz
        if not self.is_playing and self.audio_queue.qsize() >= self.prefetch_count:
            self.start_playback()

    def _play_loop(self):
        while True:
            chunk = self.audio_queue.get()
            
            if chunk is None:  # Akış sonu sinyali
                break
            
            # Gelen veriyi PyAudio stream'ine yaz (Bloklayan işlem)
            self.stream.write(chunk)
            

            if self.audio_queue.empty() and not self._finished:
                logger.warning("Ses tamponu boşaldı (Underrun). Kesinti yaşanmaması için yeniden tamponlanıyor...")
                rebuffer_target = max(2, self.prefetch_count // 3)
                

                while self.audio_queue.qsize() < rebuffer_target and not self._finished:
                    time.sleep(0.05)

    def start_playback(self):
        self.is_playing = True
        play_start_time = time.time()
        
        # Süre hesaplamaları ve loglama
        if self.request_sent_time:
            total_latency = play_start_time - self.request_sent_time
            logger.info(f"[TIMER] İstek gönderiminden ilk sesin çalınmasına kadar geçen toplam süre: {total_latency:.3f} saniye")
        
        if self.first_chunk_time:
            buffering_latency = play_start_time - self.first_chunk_time
            logger.info(f"[TIMER] İlk ses parçasının istemciye ulaşmasından çalınmaya başlanmasına kadar geçen süre (Buffer/Prefetch): {buffering_latency:.3f} saniye")
            
        # PyAudio döngüsünü ana akışı (WebSocket vb.) bloklamamak için Thread içine alıyoruz
        self.player_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.player_thread.start()

    def close(self):
        self._finished = True
        self.audio_queue.put(None)
        
        # Eğer prefetch_count'a ulaşılamadan (kısa cümle) stream bittiyse oynatmayı zorla başlat
        if not self.is_playing:
            self.start_playback()
            
        if self.player_thread:
            self.player_thread.join(timeout=5.0)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

async def stream_tts_websocket(ip, port, text, play_audio, prefetch_count):
    uri = f"ws://{ip}:{port}/v1/tts/stream"
    print(f"Sunucuya bağlanılıyor: {uri}")
    print(f"Sentezlenecek metin: {text}")
    print("Multiplexed WebSocket iletişimi bekleniyor...\n")

    player = None
    
    request_sent_time = None
    try:
        async with websockets.connect(uri) as websocket:
            print("Bağlantı kuruldu. JSON Payload gönderiliyor...")
            request_sent_time = time.time()
            await websocket.send(json.dumps({"text": text}))
            
            while True:
                try:
                    message = await websocket.recv()
                    
                    if isinstance(message, str):
                        # Text Frame: Kontrol Mesajı (JSON)
                        event_data = json.loads(message)
                        event_type = event_data.get("event")
                        
                        if event_type == "stream_start":
                            format_type = event_data.get("format", "pcm16_le")
                            print(f"\n[EVENT] Akış Başladı | Format: {format_type}")
                            
                            sample_rate = event_data.get("sample_rate", 48000)
                            
                            if play_audio and pyaudio:
                                 player = StreamingPlayer(
                                     sample_rate=sample_rate, 
                                     prefetch_count=prefetch_count, 
                                     request_sent_time=request_sent_time
                                 )
                                              
                        elif event_type == "sentence_complete":
                            print(f"\n[EVENT] Cümle sentezi tamamlandı (İndeks: {event_data.get('sentence_index')})")
                            
                        elif event_type == "stream_end":
                            print("\n[EVENT] Tüm ses akışı tamamlandı. Bağlantı kapatılıyor.")
                            await websocket.send(json.dumps({"event": "close"}))
                            break
                            
                        elif event_type == "error":
                            print(f"\n[HATA] {event_data.get('message')}")
                            break
                            
                    elif isinstance(message, bytes):
                        # Binary Frame: Ham PCM16 Ses Verisi
                        chunk_size = len(message)
                        print(f"-> Binary Chunk Alındı: {chunk_size} bytes", end="\r")
                        if player:
                            player.push_audio(message)
                            
                except websockets.ConnectionClosed:
                    print("\n[INFO] Sunucu bağlantıyı kapattı.")
                    break
    except Exception as e:
        print(f"\n[HATA] WebSocket Bağlantı Hatası: {e}")
    finally:
        if player:
            player.close()

def main():
    parser = argparse.ArgumentParser(description="Trendyol TTS Enterprise Multiplexed WebSocket İstemcisi")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Sunucu IP Adresi")
    parser.add_argument("--port", type=str, default="8000", help="Sunucu Portu")
    parser.add_argument("--text", type=str, required=True, help="Sentezlenecek akış metni")
    parser.add_argument("--play", action="store_true", help="Gelen PCM16 ses akışını anında hoparlörden çal")
    parser.add_argument("--prefetch", type=int, default=12, help="Oynatmadan önce biriktirilecek chunk sayısı")
    
    args = parser.parse_args()
    
    asyncio.run(stream_tts_websocket(args.ip, args.port, args.text, args.play, args.prefetch))

if __name__ == "__main__":
    main()
