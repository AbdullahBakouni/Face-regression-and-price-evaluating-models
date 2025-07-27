from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import tempfile
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify_faces():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 files are required.'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    
    try:
        img1.save(tmp1.name)
        img2.save(tmp2.name)

        tmp1.close()
        tmp2.close()


        result = DeepFace.verify(tmp1.name, tmp2.name)

        if result.get('verified'):
            return jsonify({'match': True, 'result': result}), 200
        else:
            return jsonify({'match': False, 'result': result}), 400

    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp1.name)
        except Exception as e:
            print(f"Failed to delete {tmp1.name}: {e}")
        try:
            os.remove(tmp2.name)
        except Exception as e:
            print(f"Failed to delete {tmp2.name}: {e}")


if __name__ == '__main__':
    app.run(host='192.168.43.167', port=5000, debug=True)
