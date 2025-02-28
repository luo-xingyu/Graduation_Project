import csv
import json
import re
import requests
import pandas as pd
import fitz
import arxiv
import pyalex
from pyalex import Works

API_KEY = "d8d0c6d8-d566-4060-8fbb-c6f517383b12"
API_URL = "https://ark.cn-beijing.volces.com/api/v3"

class Paper:
    def __init__(self, path):
        # 初始化函数，根据pdf初始化Paper对象
        self.path = path
        self.section_index = {}
        self.section_text = {}

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)  # pdf文档
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.get_paragraph()
        references_text = self.section_text.get("References", "")
        self.GetInfo(references_text)
        result_openalex = self.search_openalex_papers()
        result_arxiv = self.search_arxiv_papers()
        abstract = self.section_text['Abstract']
        conclusion = self.section_text['Conclusion']
        self.pdf.close()
        return result_openalex,result_arxiv,abstract,conclusion

    def get_paragraph(self):
        text = self.all_text
        # 删去回车和多余空格
        #text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        section_list = ['Abstract',
                        'Index Terms',
                        'Keywords',
                        'Introduction',
                        'Conclusion',
                        'References']

        # 初始化一个字典来存储找到的章节和它们在文档中出现的位置
        section_index = {}
        # 查询位置
        for section_name in section_list:
            # 将章节名称转换成大写形式
            section_name_upper = section_name.upper()
            # 查找关键词的位置(大写)
            keyword_index = text.find(section_name_upper)
            if keyword_index != -1:
                # 提取关键词后的内容
                section_index[section_name] = keyword_index
            else:
                # 查找关键词的位置(小写)
                keyword_index = text.find(section_name)
                if keyword_index != -1:
                    section_index[section_name] = keyword_index

        # 已获得所有找到的章节名称及它们在文档中出现的页码
        # print(section_index)

        # 初始化一个字典来存储找到的章节和相应内容
        section_text = {}
        start = 0  # 开始索引从0开始
        cur = 'title'
        # 获取章节内容
        for index, section_name in enumerate(section_index):
            end = section_index[section_name]
            if index == 0:
                # 取出标题，标题应该在Abstract之前
                section_text['title'] = text[start:end]
            elif index == len(section_index) - 1:
                # 最后一个章节的结束索引就是文本的结尾
                section_text[cur] = text[start:end]
                section_text['References'] = text[end + len(section_name):len(text)]
            else:
                section_text[cur] = text[start:end]

            start = end + len(section_name)  # 下一个章节的开始索引就是当前章节的结束索引
            # if cur in section_text:
            #    print(section_text[cur])
            cur = str(section_name)
        # print(section_text['References'])

        self.section_index = section_index
        self.section_text = section_text

    def search_arxiv_papers(self):
        match_cnt = 0
        for ref in self.references:
            authors = ','.join(ref['authors'])
            search_title = re.sub(r"[-:]+", " ", f"{authors} {ref['title']}")  # 替换 "-" 和 ":" 为空格
            #print(search_title)
            client = arxiv.Client()
            search = arxiv.Search(query=f'ti:{search_title}',max_results=1)
            if search:
                for result in client.results(search):
                    # 如果搜索到的论文的名字与formatted_title完全一样，则认为找到了
                    if ref['title'].lower() == result.title.lower():
                        match_cnt = match_cnt+1
                        #print("title same")
                        break
                    else:
                       print("title different: ",result.title)
            else:
                print("not found")

        if match_cnt > 0:
            percentage = (match_cnt / len(self.references))*100
            print(f'Papers are found {percentage:.2f}%')
            return percentage
        else:
            print('Fake Paper...')
            return 0

    def search_openalex_papers(self):
        # 配置pyalex
        pyalex.config.email = "tokaiteio.mejiromcqueen@gmail.com"
        pyalex.config.max_retries = 1
        match_cnt = 0
        for ref in self.references:
            results = Works().search(ref['title']).get()
            if results:
                #for authorship in results[0]["authorships"]:
                #    author_name = authorship["author"]["display_name"]
                # 如果搜索到的论文的名字与formatted_title完全一样，则认为找到了
                if ref['title'].lower() == results[0]['title'].lower():
                    match_cnt = match_cnt + 1
                    #print("title same")
            else:
                print("title different: ",results[0]['title'])
        if match_cnt > 0:
            percentage = (match_cnt / len(self.references))*100
            print(f'Papers are found {percentage:.2f}%')
            return percentage
        else:
            print('Fake Paper...')
            return 0

    def extract_title_from_references(self,references_content, prompt):
        """
        使用 OpenAI API 提取参考文献中的信息。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        data = {
            "model": "ep-20250227205656-hsbs4",
            "messages": [
                {"role": "system", "content": "你是一个可以提取文献信息的助手。"},
                {"role": "user", "content": prompt + references_content}
            ],
            "max_tokens": 15000,
            "temperature": 0.5
        }

        response = requests.post(f"{API_URL}/chat/completions", headers=headers, data=json.dumps(data))
        response_json = response.json()
        info = response_json['choices'][0]['message']['content'].strip()
        #print(response_json)
        print(info)
        return info


    def GetInfo(self,references_content):
        """
        提取参考文献中的作者，年份，标题。
        """
        all_titles = []
        title_prompt = "请你帮我将每一个参考文献的作者,年份,标题提取出来.注意，输出结果按照作者#年份#标题隔开，多个作者直接用;隔开，如果找不到，输出NULL，每个输出结果占一行\n"
        results = self.extract_title_from_references(references_content, title_prompt)
        results = results.split('\n')
        refs = []
        for title in results:
            parts = title.split('#')  # 第三个参数保证标题中的逗号不被分割
            # 处理作者列表
            authors = [author.strip() for author in parts[0].split(';')]
            refs.append({
                'authors': authors,
                'year': parts[1].strip(),
                'title': parts[2].strip()
            })
        self.references=refs
        #print(refs[1]['title'])
        #print(refs[0]['authors'][0])


if __name__ == '__main__':
    path = r'paper/demo2.pdf'
    paper = Paper(path=path)
    paper.parse_pdf()