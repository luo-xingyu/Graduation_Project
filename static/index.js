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
        loadingMessage: '准备中...',
        currentPage: 0,
        maxPage: 0,
        paragraphInfo: [], // 存储段落信息
        isTextMode: false  // 是否是纯文本模式
    },
    mounted() {
        // 初始化时在控制台输出，确认Vue实例正确加载
        console.log("Vue 应用已加载，初始加载消息：", this.loadingMessage);
    },
    methods: {
        // 清空输入框
        clearInput() {
            this.textInput = '';
            this.file = null;
        },
        
        uploadPDF() {
            // 获取文件输入元素的引用
            var fileInput = this.$refs.fileInput.$el.querySelector('input[type="file"]');
            var file = fileInput.files[0];

            if (file) {
                // 先设置加载消息和状态，再隐藏旧结果
                this.loadingMessage = '正在处理您的文件，请稍候...';
                this.isLoading = true;
                this.showResults = false;
                this.isTextMode = false; // 设置为PDF模式
                
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
                        // 清空输入框
                        this.clearInput();
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
            this.ppl = result['ppl'];
            
            // 处理段落信息
            if (result['paragraph_info']) {
                this.paragraphInfo = result['paragraph_info'];
                
                if (!this.isTextMode) {
                    // PDF模式：找出最大页码
                    this.maxPage = 0;
                    for (const para of this.paragraphInfo) {
                        if (para.page > this.maxPage) {
                            this.maxPage = para.page;
                        }
                    }
                    
                    // 设置当前页为第一页
                    this.currentPage = 0;
                }
                
                this.updatePageDisplay();
            }
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
                this.isTextMode = true; // 设置为文本模式
                
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
                    // 清空输入框
                    this.clearInput();
                })
                .catch(error => {
                    this.isLoading = false;
                    console.error('Error uploading text:', error);
                    alert('Upload failed. Please try again later.');
                });
            } else {
                alert('Enter some text');
            }
        },

        // 添加分页导航方法
        nextPage() {
            if (this.currentPage < this.maxPage) {
                this.currentPage++;
                this.updatePageDisplay();
            }
        },

        prevPage() {
            if (this.currentPage > 0) {
                this.currentPage--;
                this.updatePageDisplay();
            }
        },

        // 根据分数为文本添加颜色类名
        getColorForScore(score) {
            if (score >= 0 && score < 0.2) {
                return 'human-very-likely'; // 深绿色 #28A745
            } else if (score >= 0.2 && score < 0.4) {
                return 'human-likely'; // 黄绿色 #A0C82B
            } else if (score >= 0.4 && score < 0.6) {
                return 'uncertain'; // 黄色 #FFC107
            } else if (score >= 0.6 && score < 0.8) {
                return 'ai-likely'; // 橙色 #FD7E14
            } else if (score >= 0.8 && score <= 1.0) {
                return 'ai-very-likely'; // 红色 #DC3545
            }
            return 'uncertain'; // Default to uncertain
        },

        // 更新当前页面的显示内容
        updatePageDisplay() {
            // 判断显示方式
            let paragraphsToShow;
            
            if (this.isTextMode) {
                // 文本模式：显示所有段落
                paragraphsToShow = this.paragraphInfo;
            } else {
                // PDF模式：筛选当前页的段落
                paragraphsToShow = this.paragraphInfo.filter(para => para.page === this.currentPage);
            }
            
            if (paragraphsToShow.length === 0) {
                this.resultString = '<div>没有内容</div>';
                return;
            }
            
            // 生成带颜色标记的HTML
            let coloredText = '';
            for (const para of paragraphsToShow) {
                const score = para.score !== undefined ? para.score : 0.5; // 如果没有分数则默认为0.5
                const colorClass = this.getColorForScore(score);
                coloredText += `<div class="${colorClass}">${para.text}</div><br>`;
            }
            
            this.resultString = coloredText;
        }
    }
});
