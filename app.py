"""
Cyberbullying Detection Web Interface
Flask-based API for the detection system
"""

from flask import Flask, render_template, request, jsonify
from cyberbullying_detector import CyberbullyingDetector
from encrypted_reports import EncryptedReportsHandler
import logging
from datetime import datetime

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize components
detector = CyberbullyingDetector()
reports_handler = EncryptedReportsHandler()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_cyberbullying():
    """API endpoint for cyberbullying detection"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text field required'}), 400
        
        # Perform detection
        result = detector.detect_cyberbullying(text)
        
        # Log the detection
        logger.info(f"Detection performed: {result['severity']}")
        
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def submit_report():
    """Submit an encrypted cyberbullying report"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        severity = data.get('severity', 'unknown')
        confidence = data.get('confidence', 0.0)
        reporter_id = data.get('reporter_id', None)
        
        # Create encrypted report
        encrypted_report = reports_handler.create_report(
            text=text,
            severity=severity,
            confidence=confidence,
            reporter_id=reporter_id
        )
        
        # Store report
        storage_path = reports_handler.store_report(encrypted_report)
        
        logger.info(f"Report submitted: {encrypted_report['id']}")
        
        return jsonify({
            'success': True,
            'report_id': encrypted_report['id'],
            'stored_at': storage_path
        }), 201
    except Exception as e:
        logger.error(f"Report submission error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/<report_id>', methods=['GET'])
def retrieve_report(report_id):
    """Retrieve and decrypt a report"""
    try:
        decrypted_report = reports_handler.retrieve_report(report_id)
        
        if not decrypted_report:
            return jsonify({'error': 'Report not found'}), 404
        
        logger.info(f"Report retrieved: {report_id}")
        
        return jsonify({
            'success': True,
            'data': decrypted_report
        }), 200
    except Exception as e:
        logger.error(f"Report retrieval error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """Batch detection for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Texts array required'}), 400
        
        results = detector.batch_detect(texts)
        
        logger.info(f"Batch detection performed on {len(texts)} texts")
        
        return jsonify({
            'count': len(results),
            'results': results
        }), 200
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Cyberbullying Detection System'
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
