import json
import re
from re import search
import requests
import pandas as pd
import fitz
import arxiv
import pyalex
from pyalex import Works
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time,random
import requests
import jieba
from difflib import SequenceMatcher
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ChromeOptions
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

#API_KEY = "d8d0c6d8-d566-4060-8fbb-c6f517383b12"
#API_URL = "https://ark.cn-beijing.volces.com/api/v3"
API_KEY = "sk-QXFhTtcE8Eo9NpA47501AdC0Da3840Ee96DfDc693400B9Ec" # gpt-4o
API_URL = "https://apione.zen-x.com.cn/api/v1"
class Paper:
    def __init__(self, path):
        # 初始化函数，根据pdf初始化Paper对象
        self.path = path
        self.section_index = {}
        self.section_text = {}
        # 配置pyalex
        pyalex.config.email = "tokaiteio.mejiromcqueen@gmail.com"
        pyalex.config.max_retries = 1

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)  # pdf文档
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        references_text=self.get_references()
        # 超过2000个字符只取前2000个字符，因为只会判断前10条参考文献
        if len(references_text)>2000:
            references_text = references_text[:2000]
        self.extract_title_from_references(references_text)
        result = self.search_papers()
        self.pdf.close()
        return result

    def get_references(self):
        """提取References之后的文字并返回"""
        text = self.all_text
        # 删去多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 查找References关键词的位置
        # 先尝试查找大写形式
        ref_index = text.find("References")
        # 如果没找到大写形式，再尝试查找首字母大写形式
        if ref_index == -1:
            ref_index = text.find("REFERENCES")
        
        # 如果找到了References关键词
        if ref_index != -1:
            # 提取References之后的所有文本
            references_text = text[ref_index + len("References"):].strip()
            return references_text
        else:
            # 如果没有找到References关键词，返回空字符串或None
            return None

    def search_papers(self):
        # 用于存储线程结果的变量
        match_cnt = 0
        total_refs = len(self.references)
        results_lock = threading.Lock()  # 用于保护共享资源
        
        # 创建处理单个参考文献的函数
        def process_reference(ref):
            result = False
            ref_info = None
            
            if ref['type'] == "En":
                # 处理英文文献
                if self.search_openalex_papers(ref['title']) == True:
                    ref_info = ("OpenAlex found:", ref['title'])
                    result = True
                elif self.search_arxiv_papers(ref['authors'], ref['title']):
                    ref_info = ("arxiv found:", ref['title'])
                    result = True
            elif ref['type'] == "Cn" and re.search(r'[\u4e00-\u9fff]', ref['title']):
                # 处理中文文献
                ref_title = ref['title'].replace(" ", "")  # 删除中文标题之间的空格
                if self.search_cnki_papers(ref_title):
                    ref_info = ("cnki found:", ref_title)
                    result = True
                elif self.search_cqvip_papers(ref_title):
                    ref_info = ("cqvip found:", ref_title)
                    result = True
            
            return result, ref_info
        
        # 使用线程池并行处理引用
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_ref = {executor.submit(process_reference, ref): ref for ref in self.references}
            print(len(future_to_ref))
            # 处理完成的任务
            for future in as_completed(future_to_ref):
                try:
                    result, ref_info = future.result()
                    
                    # 线程安全地更新计数器和打印信息
                    with results_lock:
                        if result==True:
                            match_cnt += 1
                        else:
                            print("Not found")
                        if ref_info:
                            print(ref_info[0], ref_info[1])
                except Exception as e:
                    print(f"处理参考文献时出错: {e}")
        
        # 返回结果
        if match_cnt > 0:
            percentage = (match_cnt / total_refs) * 100
            print(f'Papers are found {percentage:.2f}%')
            return match_cnt / total_refs
        else:
            print('Fake Paper...')
            return 0

    def search_arxiv_papers(self,authors,title):
        authors = ','.join(authors)
        search_title = re.sub(r"[-:]+", " ", f"{authors} {title}")  # 替换 "-" 和 ":" 为空格
        #print(search_title)
        client = arxiv.Client()
        search = arxiv.Search(query=f'ti:{search_title}',max_results=1)
        if search:
            for result in client.results(search):
                # 如果搜索到的论文的名字与ref['title']完全一样，则认为找到了
                if self.text_similarity(title.lower(),result.title.lower()) :
                    return True
                else:
                   #print("arxiv title different:",title.lower(),"#",result.title.lower())
                   return False
        else:
            #print("arxiv not found", title)
            return False

    def search_openalex_papers(self,title):
        results = Works().search(title).select(["title"]).get(per_page=1)
        if results:
            #for authorship in results[0]["authorships"]:
            #    author_name = authorship["author"]["display_name"]
            # 如果搜索到的论文的名字与title完全一样，则认为找到了
            if self.text_similarity(title.lower(),results[0]['title'].lower()) :
                return True
            else:
                #print("openalex title different:",title.lower(),"#",results[0]['title'].lower())
                return False
        else:
            #print("OpenAlex not found",title)
            return False

    def search_cnki_papers(self,title):
        results = self.fetch_cnki_paper_titles(title)
        if results:
            # 如果搜索到的论文的名字与title相似度超过80%，则认为找到了
            if self.text_similarity(title,results):
                return True
            else:
                return False
        else:
            return False

    def search_cqvip_papers(self,title):
        results = self.fetch_cqvip_paper_titles(title)
        if results:
            # 如果搜索到的论文的名字与title相似度超过80%，则认为找到了
            if self.text_similarity(title,results):
                return True
            else:
                return False
        else:
            return False

    def getinfo(self,references_content, prompt):
        """
        使用 OpenAI API 提取参考文献中的信息。
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        data = {
            #"model": "ep-20250327155938-zngrc",
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "你是一个可以提取文献信息的助手,注意辨别中文和英文参考文献"},
                {"role": "user", "content": prompt + references_content}
            ],
            "max_tokens": 15000,
            "temperature": 0.5
        }

        response = requests.post(f"{API_URL}/chat/completions", headers=headers, data=json.dumps(data))
        response_json = response.json()
        info = response_json['choices'][0]['message']['content'].strip()
        #print(response_json)
        #print(info)
        return info

    def extract_title_from_references(self,references_content):
        #提取参考文献中的作者，文献类型，标题。
        title_prompt = "请你帮我将每一个参考文献的作者,是英文还是中文参考文献（中文标Cn,英文标En）,标题提取出来.注意,输出结果按照作者#Cn/En#标题隔开,多个作者直接用;隔开，只返回前十条结果，每个输出结果占一行\n"
        results = self.getinfo(references_content, title_prompt)
        results = results.split('\n')
        refs = []
        for title in results:
            parts = title.split('#')
            # 处理作者列表
            authors = [author.strip() for author in parts[0].split(';')]
            refs.append({
                'authors': authors,
                'type': parts[1].strip(),
                'title': parts[2].strip()
            })
        self.references=refs
        #print(refs)
        print("需要判断的参考文献数量:",len(refs))
        #print(refs[0])
        #print(refs[0]['authors'][0])

    def fetch_cqvip_paper_titles(self,key_word):
        url = "http://218.28.6.71:81/"
        driver_path = r'data/chromedriver-win64/chromedriver.exe'
        service = Service(driver_path)
        options = ChromeOptions()
        options.page_load_strategy = "eager"  # 仅等待 DOM 加载完成，不等待图片等资源
        driver = webdriver.Chrome(service=service,options=options)
        driver.get(url)  # 访问目标网址
        # 定位搜索框并输入关键词
        try:
            search_box = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                                                '#searchKeywords'))
            )
            search_box.send_keys(key_word)
            time.sleep(random.uniform(0.2, 0.5))
            search_btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR,
                     '#btnSearch')
                )
            )
            search_btn.click()
            WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'simple-list'))
            )
            content = driver.page_source.encode('utf-8')  # 获取页面源码（可能失效）
            strainer = SoupStrainer('dl')
            dl_bf = BeautifulSoup(content, "lxml", parse_only=strainer)

            # 获取论文标题
            dt_name = dl_bf.find('dt')  # 拿到dl下的dt
            dt_name_bf = BeautifulSoup(str(dt_name), 'lxml')
            a_name = dt_name_bf.find('a')
            # get_text()是获取标签中的所有文本，包含其子标签中的文本
            title = a_name.get_text().strip()
            driver.quit()  # 关闭浏览器窗口
            # 只返回查找到的第一个标题
            return str(title)
        except Exception as e:
            print(f"发生错误: {e}")
            return False

    def fetch_cnki_paper_titles(self,key_word):
        url = "https://www.cnki.net/"
        driver_path = r'data/chromedriver-win64/chromedriver.exe'
        service = Service(driver_path)
        options = ChromeOptions()
        options.page_load_strategy = "eager"  # 仅等待 DOM 加载完成，不等待图片等资源
        driver = webdriver.Chrome(service=service,options=options)
        driver.get(url)  # 访问目标网址
        # 定位搜索框并输入关键词
        try:
            search_box = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '#txt_SearchText'))
            )
            search_box.send_keys(key_word)
            time.sleep(random.uniform(0.2, 0.5))
            search_btn = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR,
                     'body > div.wrapper > div.searchmain > div.search-tab-content.search-form.cur > div.input-box > input.search-btn')
                )
            )
            search_btn.click()
            time.sleep(random.uniform(0.2, 0.5))
            WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'result-table-list'))
            )
            content = driver.page_source.encode('utf-8')  # 获取页面源码（可能失效）
            strainer = SoupStrainer('tr')
            tr_bf = BeautifulSoup(content, "lxml", parse_only=strainer)

            # 获取论文标题
            td_name = tr_bf.find('td', class_='name')  # 拿到tr下的td
            td_name_bf = BeautifulSoup(str(td_name), 'lxml')
            a_name = td_name_bf.find('a')
            # get_text()是获取标签中的所有文本，包含其子标签中的文本
            title = a_name.get_text().strip()
            driver.quit()  # 关闭浏览器窗口
            return str(title)

        except Exception as e:
            print(f"发生错误: {e}")
            return False
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

    def text_similarity(self,text1,text2):
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
    path = r'paper\fake.pdf'
    paper = Paper(path=path)
    paper.parse_pdf()