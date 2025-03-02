from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time,random
import requests
import jieba
from difflib import SequenceMatcher
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def fetch_cnki_paper_titles(key_word):
    url = "https://www.cnki.net/"
    driver_path = r'data/chromedriver-win64/chromedriver.exe'
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)
    driver.get(url)  # 访问目标网址
    #driver.maximize_window()

    # 定位搜索框并输入关键词
    try:
        search_box = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#txt_SearchText'))
        )
        search_box.send_keys(key_word)
        time.sleep(random.uniform(0.1, 0.2))
        search_btn = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR,
                 'body > div.wrapper > div.searchmain > div.search-tab-content.search-form.cur > div.input-box > input.search-btn')
            )
        )
        search_btn.click()
        time.sleep(random.uniform(0.1, 0.2))
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'name'))
        )
        content = driver.page_source.encode('utf-8')  # 获取页面源码（可能失效）
        soup = BeautifulSoup(content, 'lxml')  # 创建BeautifulSoup解析对象
        driver.close()  # 关闭浏览器窗口

        tbody = soup.find('tbody')  # 获取tbody标签
        tbody = BeautifulSoup(str(tbody), 'lxml')  # 解析
        tr = tbody.find('tr')  # 获取tr标签，返回一个数组
        # 对每一个tr标签进行处理
        tr_bf = BeautifulSoup(str(tr), 'lxml')
        # 获取论文标题
        td_name = tr_bf.find('td', class_='name')  # 拿到tr下的td
        td_name_bf = BeautifulSoup(str(td_name), 'lxml')
        a_name = td_name_bf.find('a')
        # get_text()是获取标签中的所有文本，包含其子标签中的文本
        title = a_name.get_text().strip()
        print(title)
        # 只返回查找到的第一个标题
        return str(title)
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    finally:
        if driver:
            driver.quit()  # 确保浏览器进程终止
def fetch_cqvip_paper_titles(key_word):
    url = "http://218.28.6.71:81/"
    driver_path = r'data/chromedriver-win64/chromedriver.exe'
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)
    driver.get(url)  # 访问目标网址
    # 定位搜索框并输入关键词
    try:
        search_box = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR,
                                            '#searchKeywords'))
        )
        search_box.send_keys(key_word)
        time.sleep(random.uniform(0.1, 0.2))
        search_btn = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR,
                 '#btnSearch')
            )
        )
        search_btn.click()
        time.sleep(random.uniform(0.1, 0.2))
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'search-result-list'))
        )
        content = driver.page_source.encode('utf-8')  # 获取页面源码（可能失效）
        soup = BeautifulSoup(content, 'lxml')  # 创建BeautifulSoup解析对象
        driver.close()  # 关闭浏览器窗口
        print("close")
        tbody = soup.find('div', class_='simple-list')  # 获取simple-list标签
        tbody = BeautifulSoup(str(tbody), 'lxml')  # 解析
        dl = tbody.find('dl')  # 获取dl标签，返回一个数组
        dl_bf = BeautifulSoup(str(dl), 'lxml')
        print("dl")
        # 获取论文标题
        dt_name = dl_bf.find('dt')  # 拿到dl下的dt
        dt_name_bf = BeautifulSoup(str(dt_name), 'lxml')
        a_name = dt_name_bf.find('a')
        # get_text()是获取标签中的所有文本，包含其子标签中的文本
        title = a_name.get_text().strip()
        # 只返回查找到的第一个标题
        print(title)
        return str(title)
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    finally:
        if driver:
            driver.quit()  # 确保浏览器进程终止
if __name__ == '__main__':
    fetch_cnki_paper_titles("人工智能生成内容技术对知识生产与传播的影响")
    fetch_cnki_paper_titles("超越 ChatGPT: 生成式 AI 的机遇, 风险与挑战")
    fetch_cnki_paper_titles("基于LightGBM模型的文本分类研究")
    fetch_cqvip_paper_titles("基于LightGBM模型的文本分类研究")
