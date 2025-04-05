new Vue({
    el: '#app',
    vuetify: new Vuetify(),
    data: {
        file: null,
        references_ratio: null,
        text_ratio: null,
        ppl: null,
        resultString: '',
        tfidf: null,
        textInput: '',
        showResults: false,
        isLoading: false,
        loadingMessage: '准备中...'
    },
    mounted() {
        // 初始化时在控制台输出，确认Vue实例正确加载
        console.log("Vue 应用已加载，初始加载消息：", this.loadingMessage);
    },
    methods: {
        uploadPDF() {
            // 获取文件输入元素的引用
            var fileInput = this.$refs.fileInput.$el.querySelector('input[type="file"]');
            var file = fileInput.files[0];

            if (file) {
                // 先设置加载消息和状态，再隐藏旧结果
                this.loadingMessage = '正在处理您的文件，请稍候...';
                this.isLoading = true;
                this.showResults = false;
                
                console.log("正在上传PDF文件，加载状态：", this.isLoading);
                console.log("加载消息：", this.loadingMessage);
                
                // 强制DOM更新
                this.$nextTick(() => {
                    console.log("DOM已更新，加载状态和消息应该可见");
                });
                
                var formData = new FormData();
                formData.append('file', file);
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/upload', true);
                xhr.onload = () => {
                    this.isLoading = false;
                    if (xhr.status == 200) {
                        // 处理上传结果
                        var result = JSON.parse(xhr.responseText);
                        this.handleUploadResult(result);
                        this.showResults = true;
                    } else {
                        alert('Upload Failed');
                    }
                };
                xhr.onerror = () => {
                    this.isLoading = false;
                    alert('Upload Failed. Network error occurred.');
                };
                xhr.send(formData);
            } else {
                alert('Choose the File');
            }
        },

        handleUploadResult(result) {
            // 更新Vue实例中的数据
            this.references_ratio = (result['rate'] * 100).toFixed(2) + "%";
            this.text_ratio = (result['text_ratio'] * 100).toFixed(2) + "%";
            this.ppl = result['ppl']
            //this.tfidf = (result['abstract_ratio_lr'] * 100).toFixed(6) + "%";
            // this.conclusionJudged = result['ppl2'] ? "HUMAN (avg_ppl>=85)" : "AI (avg_ppl<85)";

            /* var text = ""
            for (var i = 0; i < result['text1'].length; i++) {
                if(result['abstract_ppl'][i]<80){
                    text += "<span style='background-color: yellow;'>" + result['text1'][i] + "</span>";
                }
                else{
                    text += "<span>" + result['text1'][i] + "</span>";
                }
            }
            this.resultString = text; */
        },

        uploadText() {
            // 获取文本框中的内容
            var textInput = this.textInput;
        
            // 确保输入不为空
            if (textInput.trim() !== '') {
                // 先设置加载消息和状态，再隐藏旧结果
                this.loadingMessage = '正在分析文本，请稍候...';
                this.isLoading = true;
                this.showResults = false;
                
                console.log("正在上传文本，加载状态：", this.isLoading);
                console.log("加载消息：", this.loadingMessage);
                
                // 强制DOM更新
                this.$nextTick(() => {
                    console.log("DOM已更新，加载状态和消息应该可见");
                });
                
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
                    this.isLoading = false;
                    this.handleUploadResult(result);
                    this.showResults = true;
                })
                .catch(error => {
                    this.isLoading = false;
                    console.error('Error uploading text:', error);
                    alert('Upload failed. Please try again later.');
                });
            } else {
                alert('Enter some text');
            }
        }
    }
});
