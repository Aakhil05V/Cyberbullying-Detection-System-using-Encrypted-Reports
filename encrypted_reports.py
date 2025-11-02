"""
Encrypted Reports Handler
Manages encryption/decryption of cyberbullying reports for privacy
"""

from cryptography.fernet import Fernet
import base64
import json
from datetime import datetime
import os


class EncryptedReportsHandler:
    """Handler for encrypted cyberbullying reports"""

    def __init__(self, key_path='keys/encryption.key'):
        self.key_path = key_path
        self.cipher = None
        self.load_or_create_key()

    def load_or_create_key(self):
        """Load existing key or create a new encryption key"""
        try:
            if os.path.exists(self.key_path):
                with open(self.key_path, 'rb') as f:
                    key = f.read()
                    self.cipher = Fernet(key)
            else:
                self.generate_new_key()
        except Exception as e:
            print(f'Error loading key: {e}')
            self.generate_new_key()

    def generate_new_key(self):
        """Generate a new encryption key"""
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
        with open(self.key_path, 'wb') as f:
            f.write(key)
        self.cipher = Fernet(key)

    def encrypt_report(self, report_data):
        """Encrypt a cyberbullying report"""
        try:
            # Convert report to JSON string
            report_json = json.dumps(report_data)
            # Encrypt the JSON
            encrypted = self.cipher.encrypt(report_json.encode())
            return encrypted.decode()
        except Exception as e:
            print(f'Encryption error: {e}')
            return None

    def decrypt_report(self, encrypted_data):
        """Decrypt a cyberbullying report"""
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            report_data = json.loads(decrypted.decode())
            return report_data
        except Exception as e:
            print(f'Decryption error: {e}')
            return None

    def create_report(self, text, severity, confidence, reporter_id=None):
        """Create and encrypt a report"""
        report = {
            'text': text,
            'severity': severity,
            'confidence': confidence,
            'reporter_id': reporter_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        encrypted_report = self.encrypt_report(report)
        return {
            'id': self.generate_report_id(),
            'encrypted_data': encrypted_report,
            'created_at': datetime.now().isoformat()
        }

    def generate_report_id(self):
        """Generate unique report ID"""
        import uuid
        return str(uuid.uuid4())

    def store_report(self, report, storage_path='reports/'):
        """Store encrypted report to file"""
        try:
            os.makedirs(storage_path, exist_ok=True)
            report_file = os.path.join(storage_path, f"report_{report['id']}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f)
            return report_file
        except Exception as e:
            print(f'Storage error: {e}')
            return None

    def retrieve_report(self, report_id, storage_path='reports/'):
        """Retrieve and decrypt a stored report"""
        try:
            report_file = os.path.join(storage_path, f"report_{report_id}.json")
            with open(report_file, 'r') as f:
                report = json.load(f)
            decrypted_data = self.decrypt_report(report['encrypted_data'])
            return decrypted_data
        except Exception as e:
            print(f'Retrieval error: {e}')
            return None

    def batch_encrypt_reports(self, reports):
        """Encrypt multiple reports"""
        encrypted_reports = []
        for report_data in reports:
            encrypted = self.encrypt_report(report_data)
            encrypted_reports.append(encrypted)
        return encrypted_reports

    def batch_decrypt_reports(self, encrypted_reports):
        """Decrypt multiple reports"""
        decrypted_reports = []
        for encrypted_data in encrypted_reports:
            decrypted = self.decrypt_report(encrypted_data)
            decrypted_reports.append(decrypted)
        return decrypted_reports


if __name__ == '__main__':
    handler = EncryptedReportsHandler()
    test_report = {
        'text': 'Test cyberbullying report',
        'severity': 'high',
        'confidence': 0.85
    }
    encrypted = handler.encrypt_report(test_report)
    print(f'Encrypted: {encrypted[:50]}...')
    decrypted = handler.decrypt_report(encrypted)
    print(f'Decrypted: {decrypted}')
