from flask import Flask, render_template, request, jsonify
import os
import get_paper
import hc3_detection
import ml_detection
import perplexity

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
        # 保存上传的PDF文件
        file.save(upload_path)

        print("Calculating Fake References Ratio ...")
        paper = get_paper.Paper(upload_path)
        rate, abstract, conclusion = paper.parse_pdf()
        
        print("Calculating Fake Abstract Ratio(roberta) ...")
        abstract_ratio = dl_detection.predict_class_probabilities(abstract)

        print("Calculating Fake Abstract Ratio(ppl) ...")
        abstract_ppl, ppl1, text1 = perplexity.predict_class_probabilities(abstract)
    
        print("Calculating Fake Abstract Ratio(TF-IDF) ...")
        abstract_ratio_lr = ml_detection.predict_class_probabilities(abstract)

        result = {
            'rate':rate,
            'ppl1':ppl1,
            'abstract_ppl':abstract_ppl,
            'text1':text1,
            'abstract_ratio':abstract_ratio,
            'abstract_ratio_lr':abstract_ratio_lr
        }
        return jsonify(result)

@app.route('/uploadText', methods=['POST'])
def upload_text():
    # 获取上传的文本
    text = request.form.get('text')

    if not text:
        return jsonify({'error': 'No text provided'})

    print("Calculating Fake Abstract Ratio(roberta) ...")
    abstract_ratio = dl_detection.predict_class_probabilities(text)

    print("Calculating Fake Abstract Ratio(ppl) ...")
    abstract_ppl, ppl1, text1 = perplexity.predict_class_probabilities(text)

    print("Calculating Fake Abstract Ratio(TF-IDF) ...")
    abstract_ratio_lr = ml_detection.predict_class_probabilities(text)

    result = {
        'rate': 0,
        'ppl1': ppl1,
        'abstract_ppl': abstract_ppl,
        'text1': text1,
        'abstract_ratio': abstract_ratio,
        'abstract_ratio_lr':abstract_ratio_lr
    }
    return jsonify(result)
    
if __name__ == '__main__':
    app.run(debug=True)
