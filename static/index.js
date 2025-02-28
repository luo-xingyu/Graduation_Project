new Vue({
    el: '#app',
    vuetify: new Vuetify(),
    data: {
        file: null,
        fakeReferences: null,
        fakeAbstract: null,
        abstractJudged: null,
        resultString: '',
        tfidf:null,

        textInput: '', // 确保这里有 textInput 属性
        showResults: false
    },
    methods: {
        uploadPDF() {
            // 获取文件输入元素的引用
            var fileInput = this.$refs.fileInput.$el.querySelector('input[type="file"]');
            // 提取文件
            var file = fileInput.files[0];

            if (file) {
                var formData = new FormData();
                formData.append('file', file);
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/upload', true);
                xhr.onload = () => {
                    if (xhr.status == 200) {
                        alert('Upload Success');
                        var result = JSON.parse(xhr.responseText);
                        // 处理上传结果
                        this.handleUploadResult(result);
                        this.showResults = true;
                    } else {
                        alert('Upload Failed');
                    }
                };
                xhr.send(formData);
            } else {
                alert('Choose the File');
            }
        },

        handleUploadResult(result) {
            // 更新Vue实例中的数据
            this.fakeReferences = (result['rate'] * 100).toFixed(2) + "%";
            this.fakeAbstract = (result['abstract_ratio'] * 100).toFixed(6) + "%";
            this.abstractJudged = result['ppl1']>85 ? "HUMAN (avg_ppl>=85):"+result['ppl1'] : "AI (avg_ppl<85):"+result['ppl1'];
            this.tfidf = (result['abstract_ratio_lr'] * 100).toFixed(6) + "%";
            // this.conclusionJudged = result['ppl2'] ? "HUMAN (avg_ppl>=85)" : "AI (avg_ppl<85)";

            var text = ""
            for (var i = 0; i < result['text1'].length; i++) {
                if(result['abstract_ppl'][i]<80){
                    text += "<span style='background-color: yellow;'>" + result['text1'][i] + "</span>";
                }
                else{
                    text += "<span>" + result['text1'][i] + "</span>";
                }
            }
            this.resultString = text;
        },

        uploadText() {
            // 获取文本框中的内容
            var textInput = this.textInput;
        
            // 确保输入不为空
            if (textInput.trim() !== '') {
                // 创建一个 FormData 对象，将文本内容放入其中
                var formData = new FormData();
                formData.append('text', textInput);
        
                // 发送 POST 请求
                fetch('/uploadText', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Upload failed');
                    }
                    return response.json();
                })
                .then(result => {
                    // 处理上传结果
                    this.handleUploadResult(result);
                    this.showResults = true;
                    alert('Upload Success');
                })
                .catch(error => {
                    console.error('Error uploading text:', error);
                    alert('Upload failed. Please try again later.');
                });
            } else {
                alert('Enter some text');
            }
        }
    }
});
