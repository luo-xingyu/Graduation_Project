from flask import Flask, render_template, request, jsonify
import os
import get_paper
from detect_paragraph import final_pdf_detect,final_text_detect
import ml_detection
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'result': 'No file provided'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'result': 'No selected file'})

    if file:
        # 设置上传文件保存的目录
        upload_path = os.path.join('paper', file.filename)
        # 如果文件已存在，则覆盖
        try:
        # 如果文件已存在，则覆盖
            with open(upload_path, 'wb') as f:
                f.write(file.read())
            print(f"File saved successfully: {upload_path}")
        except PermissionError as e:
            print({'error': f'Permission denied: {str(e)}'})
        except Exception as e:
            print({'error': f'An error occurred while saving the file: {str(e)}'})

        print("Calculating Fake References Ratio ...")
        paper = get_paper.Paper(upload_path)
        references_rate = paper.parse_pdf()
        
        print("Calculating Fake Ratio(distilled roberta,ppl,ml) ...")
        paragraph_info, avg_score, ppl,ml_score,final_score = final_pdf_detect(upload_path)

        result = {
            'rate': references_rate,
            'ppl': ppl,
            'text_ratio': avg_score,
            'ml_score': ml_score,
            'final_score': final_score,
            'paragraph_info': paragraph_info
        }
        return jsonify(result)

@app.route('/uploadText', methods=['POST'])
def upload_text():
    # 获取上传的文本
    text = request.form.get('text')

    if not text:
        return jsonify({'error': 'No text provided'})

    print("Calculating Fake Ratio(distilled roberta) ...")
    paragraph_info, avg_score, ppl,ml_score,final_score = final_text_detect(text)

    #print("Calculating Fake Abstract Ratio(TF-IDF) ...")
    #abstract_ratio_lr = ml_detection.predict_class_probabilities(text)

    result = {
        'rate': 0,
        'ppl': ppl,
        'text_ratio': avg_score,
        'ml_score': "not use",
        'final_score': final_score,
        'paragraph_info': paragraph_info
    }
    return jsonify(result)
    
if __name__ == '__main__':
    app.run(debug=True)
