import os
import requests
from zipfile import ZipFile

# 配置参数
url = "https://cdn-mineru.openxlab.org.cn/pdf/412c4e03-e73d-483e-b449-c4b5a7839015.zip"
download_folder = "./parse_paper"  # 下载文件的保存目录
extract_folder = "./parse_paper"    # 解压文件的目录
os.environ['NO_PROXY'] = "https://cdn-mineru.openxlab.org.cn"
# 下载文件
def download_zip(url, save_path):
    try:
        # 发送请求（流式下载节省内存）
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # 检查HTTP错误

        # 获取文件名
        file_name = os.path.basename(url)
        file_path = os.path.join(save_path, file_name)
        print(file_name,file_path)
        # 写入文件
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤保持连接的空白块
                    f.write(chunk)
        print(f"文件已下载到: {file_path}")
        return file_path,file_name
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return None

# 解压文件
def extract_zip(zip_path, extract_path):
    try:
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"文件已解压到: {extract_path}")
        os.remove(zip_path)
        print(f"已删除原压缩文件: {zip_path}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False

# 执行
if __name__ == "__main__":
    zip_path,file_name = download_zip(url, download_folder)
    file_name = os.path.splitext(file_name)[0]
    extract_path = os.path.join(extract_folder, file_name)
    if zip_path:
        extract_zip(zip_path, extract_path)