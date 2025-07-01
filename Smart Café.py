import cv2
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Logging sistemini ayarla - فقط الأخطاء والمعلومات المهمة
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartCafeSystem:
    """
    Müşterileri tanıyan ve siparişleri yöneten Akıllı Kafe Sistemi
    """

    def __init__(self, data_dir: str = "kafe_verileri"):
        """
        Sistemi başlat

        Args:
            data_dir: Veri depolama klasörü
        """
        self.data_dir = data_dir
        self.faces_dir = os.path.join(data_dir, "yuzler")
        self.customers_file = os.path.join(data_dir, "musteriler.json")
        self.orders_file = os.path.join(data_dir, "siparisler.json")
        self.excel_file = os.path.join(data_dir, "kafe_veritabani.xlsx")

        # Gerekli klasörleri oluştur
        self._create_directories()

        # Mevcut verileri yükle
        self.customers = self._load_customers()
        self.orders = self._load_orders()

        # Yüz tanıma modelini ayarla
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Modeli mevcut verilerle eğit
        self._train_recognizer()

        # Kamerayı ayarla
        self.camera = None

        # Sistem değişkenleri
        self.current_customer = None
        self.is_running = False
        self.last_recognized_customer = None
        self.registration_mode = False
        self.pending_customer_name = None
        self.auto_order_mode = False

        # طباعة مسارات حفظ البيانات
        print("\n" + "=" * 60)
        print("📁 Veri Kayıt Yolları:")
        print(f"   📂 Ana Klasör: {os.path.abspath(self.data_dir)}")
        print(f"   🖼️  Yüz Resimleri: {os.path.abspath(self.faces_dir)}")
        print(f"   👥 Müşteri Verileri: {os.path.abspath(self.customers_file)}")
        print(f"   🛒 Sipariş Verileri: {os.path.abspath(self.orders_file)}")
        print(f"   📊 Excel Dosyası: {os.path.abspath(self.excel_file)}")
        print("=" * 60)

        print("✅ Akıllı Kafe Sistemi başarıyla başlatıldı")

    def _create_directories(self):
        """Gerekli klasörleri oluştur"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)

    def _load_customers(self) -> Dict:
        """JSON dosyasından müşteri verilerini yükle"""
        try:
            if os.path.exists(self.customers_file):
                with open(self.customers_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Müşteri verileri yüklenirken hata: {e}")
            return {}

    def _load_orders(self) -> List[Dict]:
        """JSON dosyasından sipariş verilerini yükle"""
        try:
            if os.path.exists(self.orders_file):
                with open(self.orders_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Sipariş verileri yüklenirken hata: {e}")
            return []

    def _save_customers(self):
        """Müşteri verilerini JSON dosyasına kaydet"""
        try:
            with open(self.customers_file, 'w', encoding='utf-8') as f:
                json.dump(self.customers, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Müşteri verileri kaydedilirken hata: {e}")

    def _save_orders(self):
        """Sipariş verilerini JSON dosyasına kaydet"""
        try:
            with open(self.orders_file, 'w', encoding='utf-8') as f:
                json.dump(self.orders, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Sipariş verileri kaydedilirken hata: {e}")

    def _export_to_excel(self):
        """Verileri Excel dosyasına aktar"""
        try:
            # Müşteri verilerini hazırla
            customers_data = []
            for customer_id, customer_info in self.customers.items():
                customers_data.append({
                    'Müşteri ID': customer_id,
                    'İsim': customer_info['name'],
                    'Ziyaret Sayısı': customer_info['visit_count'],
                    'Toplam Harcama': customer_info['total_spent'],
                    'Sadakat Puanları': customer_info['loyalty_points'],
                    'Üyelik Seviyesi': customer_info['membership_level'],
                    'Kayıt Tarihi': customer_info['registration_date'],
                    'Son Ziyaret': customer_info['last_visit'],
                    'En Çok Sipariş Verilen': ', '.join(customer_info.get('favorite_items', [])[:3])
                })

            # Sipariş verilerini hazırla
            orders_data = []
            for order in self.orders:
                orders_data.append({
                    'Sipariş ID': order['order_id'],
                    'Müşteri ID': order['customer_id'],
                    'Müşteri Adı': order['customer_name'],
                    'Siparişler': ', '.join(order['items']),
                    'Toplam Tutar': order['total_amount'],
                    'Sipariş Tarihi': order['order_date']
                })

            # Excel dosyası oluştur
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                pd.DataFrame(customers_data).to_excel(writer, sheet_name='Müşteriler', index=False)
                pd.DataFrame(orders_data).to_excel(writer, sheet_name='Siparişler', index=False)

            print(f"📊 Veriler Excel dosyasına kaydedildi: {self.excel_file}")

        except Exception as e:
            logger.error(f"Excel'e veri aktarırken hata: {e}")

    def _train_recognizer(self):
        """Yüz tanıma modelini eğit"""
        try:
            faces = []
            labels = []

            for customer_id, customer_info in self.customers.items():
                customer_faces_dir = os.path.join(self.faces_dir, customer_id)
                if os.path.exists(customer_faces_dir):
                    for image_file in os.listdir(customer_faces_dir):
                        if image_file.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(customer_faces_dir, image_file)
                            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            if image is not None:
                                faces.append(image)
                                labels.append(int(customer_id))

            if len(faces) > 0:
                self.recognizer.train(faces, np.array(labels))
                print(f"🎯 Tanıma modeli {len(faces)} resimle eğitildi")
            else:
                print("⚠️ Eğitim için resim bulunamadı")

        except Exception as e:
            logger.error(f"Model eğitiminde hata: {e}")

    def _generate_customer_id(self) -> str:
        """Müşteri için benzersiz ID oluştur"""
        if not self.customers:
            return "1"
        return str(max(int(cid) for cid in self.customers.keys()) + 1)

    def _calculate_loyalty_points(self, amount: float) -> int:
        """Ödenen miktara göre sadakat puanlarını hesapla"""
        return int(amount // 5)  # Her 5 TL için 1 puan

    def _get_membership_level(self, visit_count: int, total_spent: float) -> str:
        """Ziyaret sayısı ve harcamaya göre üyelik seviyesini belirle"""
        if visit_count >= 50 or total_spent >= 1000:
            return "Altın"
        elif visit_count >= 20 or total_spent >= 500:
            return "Gümüş"
        elif visit_count >= 5 or total_spent >= 100:
            return "Bronz"
        else:
            return "Standart"

    def _get_customer_recommendations(self, customer_id: str) -> List[str]:
        """Önceki siparişlere göre içecek öner"""
        customer_orders = [order for order in self.orders if order['customer_id'] == customer_id]

        if not customer_orders:
            return []

        # Sipariş sıklığını hesapla
        item_frequency = {}
        for order in customer_orders:
            for item in order['items']:
                item_frequency[item] = item_frequency.get(item, 0) + 1

        # Siparişleri sıklığa göre sırala
        recommendations = sorted(item_frequency.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in recommendations[:5]]

    def _get_last_order(self, customer_id: str) -> Optional[Dict]:
        """Müşterinin son siparişini getir"""
        customer_orders = [order for order in self.orders if order['customer_id'] == customer_id]
        if customer_orders:
            return customer_orders[-1]
        return None

    def start_camera(self):
        """Kamerayı başlat"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Kamera başlatılamadı")
                return False

            # Kamera kalitesini iyileştir
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            print("📸 Kamera başarıyla başlatıldı")
            return True

        except Exception as e:
            logger.error(f"Kamera başlatılırken hata: {e}")
            return False

    def detect_and_recognize_face(self, frame) -> Tuple[Optional[str], Optional[str], List[Tuple[int, int, int, int]]]:
        """
        Çerçevedeki yüzleri tespit et ve tanı
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        recognized_customer_id = None
        recognized_customer_name = None

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            try:
                if len(self.customers) > 0:
                    customer_id, confidence = self.recognizer.predict(face_roi)

                    if confidence < 80:  # Güven eşiği düşürüldü
                        customer_id_str = str(customer_id)
                        if customer_id_str in self.customers:
                            recognized_customer_id = customer_id_str
                            recognized_customer_name = self.customers[customer_id_str]['name']

                            # Müşteri değiştiyse bilgi göster
                            if self.last_recognized_customer != customer_id_str:
                                self._display_customer_welcome(customer_id_str)
                                self.last_recognized_customer = customer_id_str

            except Exception as e:
                logger.error(f"Yüz tanımada hata: {e}")

        # Tanınmayan yüz varsa
        if faces.any() and not recognized_customer_id:
            if self.last_recognized_customer != "unknown":
                self.last_recognized_customer = "unknown"
                if not self.registration_mode:
                    print("\n" + "=" * 50)
                    print("👤 YENİ MÜŞTERİ TESPİT EDİLDİ!")
                    print("Kayıt için müşteri adını girin...")
                    print("=" * 50)
                    self.registration_mode = True

        elif not faces.any():
            self.last_recognized_customer = None

        return recognized_customer_id, recognized_customer_name, faces.tolist()

    def _display_customer_welcome(self, customer_id: str):
        """Tanınan müşteri için karşılama mesajı"""
        customer_info = self.get_customer_info(customer_id)
        last_order = self._get_last_order(customer_id)

        print("\n" + "=" * 60)
        print(f"🎉 HOŞGELDİNİZ {customer_info['name'].upper()}!")
        print("=" * 60)
        print(f"👑 Üyelik Seviyesi: {customer_info['membership_level']}")
        print(f"🏆 Ziyaret Sayısı: {customer_info['visit_count']}")
        print(f"⭐ Sadakat Puanları: {customer_info['loyalty_points']}")
        print(f"💰 Toplam Harcama: {customer_info['total_spent']} TL")

        if last_order:
            print(f"📋 Son Siparişiniz: {', '.join(last_order['items'])}")
            print(f"📅 Tarih: {last_order['order_date']}")

        if customer_info['recommendations']:
            print(f"🔥 En Çok Tercih Ettikleriniz: {', '.join(customer_info['recommendations'][:3])}")

        print("=" * 60)
        print("💡 Yeni sipariş vermek için 'Enter' tuşuna basın")
        print("=" * 60)

    def register_new_customer(self, frame, customer_name: str) -> str:
        """Yeni müşteri kaydet"""
        try:
            customer_id = self._generate_customer_id()

            # Müşteri için klasör oluştur
            customer_faces_dir = os.path.join(self.faces_dir, customer_id)
            os.makedirs(customer_faces_dir, exist_ok=True)

            # Yüz resmini kaydet
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # En büyük yüzü kaydet
                (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
                face_roi = gray[y:y + h, x:x + w]

                # Tanımayı iyileştirmek için yüzden birkaç resim kaydet
                for i in range(10):  # Daha fazla resim
                    face_filename = os.path.join(customer_faces_dir, f"yuz_{i}.jpg")
                    cv2.imwrite(face_filename, face_roi)

                # Ana resim olarak renkli resim kaydet
                main_image = frame[y:y + h, x:x + w]
                main_filename = os.path.join(customer_faces_dir, "ana.jpg")
                cv2.imwrite(main_filename, main_image)

                # Müşteriyi veritabanına ekle
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.customers[customer_id] = {
                    'name': customer_name,
                    'registration_date': current_time,
                    'last_visit': current_time,
                    'visit_count': 0,
                    'total_spent': 0.0,
                    'loyalty_points': 0,
                    'membership_level': 'Standart',
                    'favorite_items': []
                }

                # Verileri kaydet
                self._save_customers()
                self._train_recognizer()

                print(f"✅ YENİ MÜŞTERİ KAYDEDİLDİ: {customer_name} (ID: {customer_id})")
                print(f"📂 Resimler kaydedildi: {customer_faces_dir}")

                # Otomatik sipariş moduna geç
                self.auto_order_mode = True
                self.current_customer = customer_id

                return customer_id
            else:
                print("❌ Yüz tespit edilemedi")
                return None

        except Exception as e:
            logger.error(f"Yeni müşteri kaydında hata: {e}")
            return None

    def add_order(self, customer_id: str, items: List[str], total_amount: float):
        """Yeni sipariş ekle"""
        try:
            if customer_id not in self.customers:
                logger.error(f"Müşteri bulunamadı: {customer_id}")
                return False

            # Sipariş ID'si oluştur
            order_id = len(self.orders) + 1

            # Siparişi ekle
            order = {
                'order_id': order_id,
                'customer_id': customer_id,
                'customer_name': self.customers[customer_id]['name'],
                'items': items,
                'total_amount': total_amount,
                'order_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.orders.append(order)

            # Müşteri verilerini güncelle
            customer = self.customers[customer_id]
            customer['visit_count'] += 1
            customer['total_spent'] += total_amount
            customer['last_visit'] = order['order_date']

            # Sadakat puanlarını hesapla
            new_points = self._calculate_loyalty_points(total_amount)
            customer['loyalty_points'] += new_points

            # Üyelik seviyesini güncelle
            customer['membership_level'] = self._get_membership_level(
                customer['visit_count'],
                customer['total_spent']
            )

            # Favori ürünleri güncelle
            for item in items:
                if item not in customer['favorite_items']:
                    customer['favorite_items'].append(item)

            # Verileri kaydet
            self._save_customers()
            self._save_orders()
            self._export_to_excel()

            print(f"\n✅ SİPARİŞ KAYDEDİLDİ!")
            print(f"👤 Müşteri: {customer['name']}")
            print(f"🛒 Siparişler: {', '.join(items)}")
            print(f"💰 Tutar: {total_amount} TL")
            print(f"⭐ Kazanılan Puan: {new_points}")
            print(f"🏆 Toplam Puan: {customer['loyalty_points']}")
            print(f"👑 Üyelik: {customer['membership_level']}")

            return True

        except Exception as e:
            logger.error(f"Sipariş eklemede hata: {e}")
            return False

    def get_customer_info(self, customer_id: str) -> Optional[Dict]:
        """Müşteri bilgilerini al"""
        if customer_id in self.customers:
            customer_info = self.customers[customer_id].copy()
            customer_info['recommendations'] = self._get_customer_recommendations(customer_id)
            return customer_info
        return None

    def _handle_new_customer_order(self):
        """Yeni müşteri için sipariş al"""
        print(f"\n☕ {self.customers[self.current_customer]['name']}, bugün ne içmek istersiniz?")
        print("Örnek: Türk Kahvesi, Cappuccino, Mocha, Hot Chocolate, vb.")

        order_input = input("Siparişiniz: ").strip()
        if order_input:
            try:
                amount_input = input("Tutar (TL): ").strip()
                if amount_input:
                    total_amount = float(amount_input)
                    items = [item.strip() for item in order_input.split(',')]

                    if self.add_order(self.current_customer, items, total_amount):
                        self.auto_order_mode = False
                        self.current_customer = None
                        print("\n🎉 Teşekkürler! Siparişiniz alındı.")
                    else:
                        print("❌ Sipariş kaydetme hatası")
                else:
                    print("❌ Tutar girilmelidir")
            except ValueError:
                print("❌ Geçerli bir tutar giriniz")
        else:
            print("❌ Sipariş girilmelidir")

    def _handle_existing_customer_order(self, customer_id: str):
        """Mevcut müşteri için sipariş al"""
        print(f"\n☕ Yeni siparişinizi alabilir miyiz?")
        order_input = input("Siparişiniz (boş bırakırsanız iptal): ").strip()

        if order_input:
            try:
                amount_input = input("Tutar (TL): ").strip()
                if amount_input:
                    total_amount = float(amount_input)
                    items = [item.strip() for item in order_input.split(',')]

                    if self.add_order(customer_id, items, total_amount):
                        print("\n🎉 Teşekkürler! Siparişiniz alındı.")
                    else:
                        print("❌ Sipariş kaydetme hatası")
                else:
                    print("❌ Tutar girilmelidir")
            except ValueError:
                print("❌ Geçerli bir tutar giriniz")

    def run_system(self):
        """Ana sistemi çalıştır"""
        if not self.start_camera():
            return

        self.is_running = True
        recognized_customer = None

        print("\n" + "=" * 60)
        print("🎯 AKILLI KAFE SİSTEMİ")
        print("=" * 60)
        print("📱 Sistem otomatik çalışmaktadır")
        print("🔄 Müşteri tespiti ve sipariş alma otomatik")
        print("❌ Çıkmak için 'q' tuşuna basın")
        print("=" * 60)

        while self.is_running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                # Yüzleri tespit et ve tanı
                customer_id, customer_name, faces = self.detect_and_recognize_face(frame)

                # Yüzlerin etrafına çerçeve çiz
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if customer_name:
                        cv2.putText(frame, customer_name, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        recognized_customer = customer_id
                    else:
                        cv2.putText(frame, "Bilinmiyor", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Sistem bilgilerini göster
                cv2.putText(frame, f"Kayitli Musteriler: {len(self.customers)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Toplam Siparisler: {len(self.orders)}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if recognized_customer:
                    customer_info = self.get_customer_info(recognized_customer)
                    cv2.putText(frame, f"Ziyaretler: {customer_info['visit_count']}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Puanlar: {customer_info['loyalty_points']}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.imshow('Akilli Kafe Sistemi', frame)

                # Klavye kontrolleri
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == 13:  # Enter tuşu
                    if self.auto_order_mode and self.current_customer:
                        self._handle_new_customer_order()
                    elif recognized_customer:
                        self._handle_existing_customer_order(recognized_customer)
                    elif self.registration_mode:
                        customer_name = input("\nYeni müşteri adı: ").strip()
                        if customer_name:
                            new_customer_id = self.register_new_customer(frame, customer_name)
                            if new_customer_id:
                                recognized_customer = new_customer_id
                                self.registration_mode = False
                            else:
                                print("❌ Kayıt başarısız")
                        else:
                            print("❌ İsim girilmelidir")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Sistem çalıştırılırken hata: {e}")

        self._cleanup()

    def _cleanup(self):
        """Kapatma sırasında kaynakları temizle"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("🔒 Sistem kapatıldı")


# Sistemi çalıştır
if __name__ == "__main__":
    try:
        # Kafe sistemini oluştur
        cafe_system = SmartCafeSystem()

        # Sistemi çalıştır
        cafe_system.run_system()

    except Exception as e:
        logger.error(f"Sistem çalıştırılırken hata: {e}")
        print("❌ Sistem hatası. Lütfen gereksinimlerinizi kontrol edin.")