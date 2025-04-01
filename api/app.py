from flask import Flask, jsonify, request
from flask_cors import CORS  # Import Flask-CORS
from generate_audit_finding import return_audit_findings
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def hello():
    return "Hello, World!"

# @app.route('/api/classify-comment', methods=['POST'])
# def classify_comments():
#     try:
#         req_body = request.get_json()
        
#         # final_df = ac.compute_compliance()
        
#         # Convert DataFrame to JSON string
#         json_data = "hello world"

#         return jsonify(json_data)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500  # Return error with appropriate status code

@app.route('/api/get-audit-findings', methods=['POST'])
def get_audit_findings():
    try:
        
        req_body = request.get_json()
        
        try:
            audit_finding = return_audit_findings(req_body)
        except Exception as e:
            print(traceback.format_exception())
        
        # Convert DataFrame to JSON string
        json_data = audit_finding

        return jsonify(json_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error with appropriate status code

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production