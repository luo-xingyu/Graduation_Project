import requests
import time,os
from zipfile import ZipFile
from zip import download_zip,extract_zip
Authorization = "Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIyMTAwNTk1NCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0MzM4NTMxMCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTkzNzA2MDg2NjAiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJjZGNhNTFkMi0wYWU0LTQzYWItYTkwNy1lOTJmOTQzMzMwMDQiLCJlbWFpbCI6IiIsImV4cCI6MTc0NDU5NDkxMH0.0r5zcQK8ZzDpud11sgtq-Ws7ih9fQo0S_QR43UV8cDemKnT7CaT4HkhbIQHvNxZRdRei-Id4NP3Me7KBy56T9w"
header = {
    'Content-Type':'application/json',
    "Authorization": Authorization
}
def parse_paper(file_path):
    url = 'https://mineru.net/api/v4/file-urls/batch'
    file_name=os.path.basename(file_path)
    print(file_name)
    data = {
        "enable_formula": False,
        "language": "en",
        "layout_model":"doclayout_yolo",
        "enable_table": True,
        "files": [
            {"name":file_name, "is_ocr":True,"data_id": "abcd"}
        ]
    }
    try:
        response = requests.post(url,headers=header,json=data)
        if response.status_code == 200:
            result = response.json()
            print('response success. result:{}'.format(result))
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                print('batch_id:{},urls:{}'.format(batch_id, urls))
                with open(file_path, 'rb') as f:
                    res_upload = requests.put(urls[0], data=f)
                if res_upload.status_code == 200:
                    print("upload success")
                else:
                    print("upload failed")
            else:
                print('apply upload url failed,reason:{}'.format(result.msg))
        else:
            print('response not success. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)

    url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
    start_time = time.time()
    res = requests.get(url, headers=header, timeout=1)
    while res.json()['data']['extract_result'][0]['state'] != 'done':
        time.sleep(1)
        res = requests.get(url, headers=header, timeout=1)
        print(res.json())
    zip_path = res.json()['data']['extract_result'][0]['full_zip_url']
    print(zip_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"耗时: {elapsed_time:.2f} 秒")
    result = unzip(zip_path,file_name)
    print(result)
    return result

def unzip(url,newname):
    download_folder = "./parse_paper"  # 下载文件的保存目录
    extract_folder = "./parse_paper"  # 解压文件的目录
    os.environ['NO_PROXY'] = "https://cdn-mineru.openxlab.org.cn" #禁止使用代理
    zip_path, file_name = download_zip(url, download_folder)
    file_name = os.path.splitext(file_name)[0] #去掉后缀.zip
    extract_path = os.path.join(extract_folder, file_name)
    print("extract_path:",extract_path)
    result = os.path.join(extract_path, file_name+"_content_list.json")
    if zip_path:
        newname = os.path.splitext(newname)[0] #去掉后缀.pdf
        extract_zip(zip_path, extract_path,newname)
        return result

if __name__ == "__main__":
    file_path = r"paper\Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper.pdf"
    parse_paper(file_path)