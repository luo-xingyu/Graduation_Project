import csv
import json
import re
import requests
import pandas as pd
import fitz
import arxiv
import pyalex
from pyalex import Works
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import requests
import csv
import jieba
from difflib import SequenceMatcher
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from test import text_similarity

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
        self.search_cnki_papers()
        return
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
                    # 如果搜索到的论文的名字与ref['title']完全一样，则认为找到了
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
                # 如果搜索到的论文的名字与ref['title']完全一样，则认为找到了
                if ref['title'].lower() == results[0]['title'].lower():
                    match_cnt = match_cnt + 1
                    #print("title same")
                else:
                    print("title different: ",results[0]['title'])
            else:
                print("not found")
        if match_cnt > 0:
            percentage = (match_cnt / len(self.references))*100
            print(f'Papers are found {percentage:.2f}%')
            return percentage
        else:
            print('Fake Paper...')
            return 0

    def search_cnki_papers(self):
        match_cnt = 0
        for ref in self.references:
            results = self.fetch_cnki_paper_titles(ref['title'])
            if results:
                # 如果搜索到的论文的名字与ref['title']相似度超过80%，则认为找到了
                if text_similarity(ref['title'],results):
                    match_cnt = match_cnt + 1
                    print("title same")
                else:
                    print("title different: ",results,ref['title'])
            else:
                print("not found")
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

    def fetch_cnki_paper_titles(self,key_word):
        url = "https://www.cnki.net/"
        driver_path = r'data/chromedriver-win64/chromedriver.exe'
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service)
        driver.get(url)  # 访问目标网址
        driver.maximize_window()

        # 定位搜索框并输入关键词
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#txt_SearchText'))
        )
        search_box.send_keys(key_word)

        search_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR,
                 'body > div.wrapper > div.searchmain > div.search-tab-content.search-form.cur > div.input-box > input.search-btn')
            )
        )
        search_btn.click()
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'name'))
        )
        content = driver.page_source.encode('utf-8')  # 获取页面源码（可能失效）
        soup = BeautifulSoup(content, 'lxml')  # 创建BeautifulSoup解析对象
        driver.close()  # 关闭浏览器窗口

        tbody = soup.find_all('tbody')  # 获取tbody标签
        tbody = BeautifulSoup(str(tbody[0]), 'lxml')  # 解析
        tr = tbody.find_all('tr')  # 获取tr标签，返回一个数组
        # 对每一个tr标签进行处理
        for item in tr:
            tr_bf = BeautifulSoup(str(item), 'lxml')
            # 获取论文标题
            td_name = tr_bf.find_all('td', class_='name')  # 拿到tr下的td
            td_name_bf = BeautifulSoup(str(td_name[0]), 'lxml')
            a_name = td_name_bf.find_all('a')
            # get_text()是获取标签中的所有文本，包含其子标签中的文本
            title = a_name[0].get_text().strip()
            # 只返回查找到的第一个标题
            return str(title)
            """
            # 获取包含作者的那个td
            td_author = tr_bf.find_all('td', class_='author')
            td_author_bf = BeautifulSoup(str(td_author), 'lxml')
            # 每个a标签中都包含了一个作者名
            a_author = td_author_bf.find_all('a')
            authors = []
            # 拿到每一个a标签里的作者名
            for author in a_author:
                name = author.get_text().strip()  # 获取学者的名字
                #print('name : ' + name)
                authors.append(name)
            print(title,authors)
            """

    def text_similarity(text1, text2):
        # 计算两段文本的相似度（0到1之间）。
        # 使用 jieba 分词
        words1 = " ".join(jieba.cut(text1))
        words2 = " ".join(jieba.cut(text2))
        # 创建 SequenceMatcher 对象
        matcher = SequenceMatcher(None, words1, words2)
        # 返回相似度
        if matcher.ratio() >= 0.8:
            return True
        return False


if __name__ == '__main__':
    path = r'paper/demo3.pdf'
    paper = Paper(path=path)
    paper.parse_pdf()