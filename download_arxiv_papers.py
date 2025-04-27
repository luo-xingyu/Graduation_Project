#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载arXiv论文脚本
该脚本使用arXiv API从不同学科领域下载论文，并保存到arxivpaper文件夹中
为每个分类创建对应的文件夹，并更新JSON文件来跟踪下载进度

用法:
    python download_arxiv_papers.py [--total 1000]

参数:
    --total: 要下载的论文总数，默认为1000
"""

import os
import time
import random
import arxiv
import requests
import argparse
import signal
import sys
import json
from tqdm import tqdm
from pathlib import Path

# 确保输出目录存在
OUTPUT_DIR = Path("arxivpaper")
OUTPUT_DIR.mkdir(exist_ok=True)

# 进度文件路径
PROGRESS_FILE = OUTPUT_DIR / "download_progress.json"

# 创建arXiv客户端
client = arxiv.Client()

# arXiv主要分类
CATEGORIES = [
    # 计算机科学 (Computer Science)
    "cs.AI",  # 人工智能 (Artificial Intelligence)
    "cs.CL",  # 计算语言学 (Computation and Language)
    "cs.CC",  # 计算复杂性 (Computational Complexity)
    "cs.CE",  # 计算工程、金融与科学 (Computational Engineering, Finance, and Science)
    "cs.CG",  # 计算几何 (Computational Geometry)
    "cs.GT",  # 计算机游戏理论 (Computer Science and Game Theory)
    "cs.CV",  # 计算机视觉与模式识别 (Computer Vision and Pattern Recognition)
    "cs.CY",  # 计算机与社会 (Computers and Society)
    "cs.CR",  # 密码学与安全 (Cryptography and Security)
    "cs.DS",  # 数据结构与算法 (Data Structures and Algorithms)
    "cs.DB",  # 数据库 (Databases)
    "cs.DL",  # 数字图书馆 (Digital Libraries)
    "cs.DM",  # 离散数学 (Discrete Mathematics)
    "cs.DC",  # 分布式、并行与集群计算 (Distributed, Parallel, and Cluster Computing)
    "cs.ET",  # 新兴技术 (Emerging Technologies)
    "cs.FL",  # 形式语言与自动机理论 (Formal Languages and Automata Theory)
    "cs.GL",  # 一般文学 (General Literature)
    "cs.GR",  # 图形学 (Graphics)
    "cs.AR",  # 硬件架构 (Hardware Architecture)
    "cs.HC",  # 人机交互 (Human-Computer Interaction)
    "cs.IR",  # 信息检索 (Information Retrieval)
    "cs.IT",  # 信息理论 (Information Theory)
    "cs.LG",  # 机器学习 (Machine Learning)
    "cs.LO",  # 逻辑与计算机科学 (Logic in Computer Science)
    "cs.MS",  # 数学软件 (Mathematical Software)
    "cs.MA",  # 多智能体系统 (Multiagent Systems)
    "cs.MM",  # 多媒体 (Multimedia)
    "cs.NI",  # 网络与互联网架构 (Networking and Internet Architecture)
    "cs.NE",  # 神经与进化计算 (Neural and Evolutionary Computing)
    "cs.NA",  # 数值分析 (Numerical Analysis)
    "cs.OS",  # 操作系统 (Operating Systems)
    "cs.OH",  # 其他计算机科学 (Other Computer Science)
    "cs.PF",  # 性能 (Performance)
    "cs.PL",  # 编程语言 (Programming Languages)
    "cs.RO",  # 机器人学 (Robotics)
    "cs.SI",  # 社交与信息网络 (Social and Information Networks)
    "cs.SE",  # 软件工程 (Software Engineering)
    "cs.SD",  # 声音 (Sound)
    "cs.SC",  # 符号计算 (Symbolic Computation)
    "cs.SY",  # 系统与控制 (Systems and Control)

    # 数学 (Mathematics)
    "math.AG",  # 代数几何 (Algebraic Geometry)
    "math.AT",  # 代数拓扑 (Algebraic Topology)
    "math.AP",  # 分析与偏微分方程 (Analysis of PDEs)
    "math.CT",  # 范畴论 (Category Theory)
    "math.CA",  # 经典分析与ODE (Classical Analysis and ODEs)
    "math.CO",  # 组合学 (Combinatorics)
    "math.AC",  # 交换代数 (Commutative Algebra)
    "math.CV",  # 复变函数 (Complex Variables)
    "math.DG",  # 微分几何 (Differential Geometry)
    "math.DS",  # 动力系统 (Dynamical Systems)
    "math.FA",  # 泛函分析 (Functional Analysis)
    "math.GM",  # 一般数学 (General Mathematics)
    "math.GN",  # 一般拓扑 (General Topology)
    "math.GT",  # 几何拓扑 (Geometric Topology)
    "math.GR",  # 群论 (Group Theory)
    "math.HO",  # 数学历史与概述 (History and Overview)
    "math.IT",  # 信息理论 (Information Theory)
    "math.KT",  # K理论与同调 (K-Theory and Homology)
    "math.LO",  # 逻辑 (Logic)
    "math.MP",  # 数学物理 (Mathematical Physics)
    "math.MG",  # 度量几何 (Metric Geometry)
    "math.NT",  # 数论 (Number Theory)
    "math.NA",  # 数值分析 (Numerical Analysis)
    "math.OA",  # 算子代数 (Operator Algebras)
    "math.OC",  # 优化与控制 (Optimization and Control)
    "math.PR",  # 概率 (Probability)
    "math.QA",  # 量子代数 (Quantum Algebra)
    "math.RT",  # 表示论 (Representation Theory)
    "math.RA",  # 环与代数 (Rings and Algebras)
    "math.SP",  # 谱理论 (Spectral Theory)
    "math.ST",  # 统计理论 (Statistics Theory)
    "math.SG",  # 辛几何 (Symplectic Geometry)

    # 物理 (Physics)
    "astro-ph.CO",  # 宇宙学与河外天文学 (Cosmology and Nongalactic Astrophysics)
    "astro-ph.EP",  # 地球与行星天体物理学 (Earth and Planetary Astrophysics)
    "astro-ph.GA",  # 银河系天体物理学 (Astrophysics of Galaxies)
    "astro-ph.HE",  # 高能天体物理学 (High Energy Astrophysical Phenomena)
    "astro-ph.IM",  # 仪器与方法 (Instrumentation and Methods for Astrophysics)
    "astro-ph.SR",  # 太阳与恒星天体物理学 (Solar and Stellar Astrophysics)
    "cond-mat.dis-nn",  # 无序系统与神经网络 (Disordered Systems and Neural Networks)
    "cond-mat.mes-hall",  # 介观系统与量子霍尔效应 (Mesoscale and Nanoscale Physics)
    "cond-mat.mtrl-sci",  # 材料科学 (Materials Science)
    "cond-mat.other",  # 其他凝聚态物理 (Other Condensed Matter)
    "cond-mat.quant-gas",  # 量子气体 (Quantum Gases)
    "cond-mat.soft",  # 软物质 (Soft Condensed Matter)
    "cond-mat.stat-mech",  # 统计力学 (Statistical Mechanics)
    "cond-mat.str-el",  # 强关联电子系统 (Strongly Correlated Electrons)
    "cond-mat.supr-con",  # 超导 (Superconductivity)
    "gr-qc",  # 广义相对论与量子宇宙学 (General Relativity and Quantum Cosmology)
    "hep-ex",  # 高能物理实验 (High Energy Physics - Experiment)
    "hep-lat",  # 高能物理格点 (High Energy Physics - Lattice)
    "hep-ph",  # 高能物理现象学 (High Energy Physics - Phenomenology)
    "hep-th",  # 高能物理理论 (High Energy Physics - Theory)
    "math-ph",  # 数学物理 (Mathematical Physics)
    "nlin.AO",  # 适应与自组织 (Adaptation and Self-Organizing Systems)
    "nlin.CG",  # 细胞自动机与格点气体 (Cellular Automata and Lattice Gases)
    "nlin.CD",  # 混沌动力学 (Chaotic Dynamics)
    "nlin.PS",  # 模式形成与孤子 (Pattern Formation and Solitons)
    "nlin.SI",  # 完全可积系统 (Exactly Solvable and Integrable Systems)
    "nucl-ex",  # 核实验 (Nuclear Experiment)
    "nucl-th",  # 核理论 (Nuclear Theory)
    "physics.acc-ph",  # 加速器物理 (Accelerator Physics)
    "physics.app-ph",  # 应用物理 (Applied Physics)
    "physics.ao-ph",  # 大气与海洋物理 (Atmospheric and Oceanic Physics)
    "physics.atom-ph",  # 原子物理 (Atomic Physics)
    "physics.atm-clus",  # 原子与分子团簇 (Atomic and Molecular Clusters)
    "physics.bio-ph",  # 生物物理 (Biological Physics)
    "physics.chem-ph",  # 化学物理 (Chemical Physics)
    "physics.class-ph",  # 经典物理 (Classical Physics)
    "physics.comp-ph",  # 计算物理 (Computational Physics)
    "physics.data-an",  # 数据分析、统计与概率 (Data Analysis, Statistics and Probability)
    "physics.flu-dyn",  # 流体动力学 (Fluid Dynamics)
    "physics.gen-ph",  # 一般物理 (General Physics)
    "physics.geo-ph",  # 地球物理 (Geophysics)
    "physics.hist-ph",  # 物理学史与哲学 (History and Philosophy of Physics)
    "physics.ins-det",  # 仪器与探测器 (Instrumentation and Detectors)
    "physics.med-ph",  # 医学物理 (Medical Physics)
    "physics.optics",  # 光学 (Optics)
    "physics.ed-ph",  # 物理教育 (Physics Education)
    "physics.soc-ph",  # 社会物理与社会网络 (Physics and Society)
    "physics.plasm-ph",  # 等离子体物理 (Plasma Physics)
    "physics.pop-ph",  # 大众物理 (Popular Physics)
    "physics.space-ph",  # 空间物理 (Space Physics)
    "quant-ph",  # 量子物理 (Quantum Physics)

    # 生物学 (Quantitative Biology)
    "q-bio.BM",  # 生物分子 (Biomolecules)
    "q-bio.CB",  # 细胞行为 (Cell Behavior)
    "q-bio.GN",  # 基因组学 (Genomics)
    "q-bio.MN",  # 分子网络 (Molecular Networks)
    "q-bio.NC",  # 神经元与认知 (Neurons and Cognition)
    "q-bio.OT",  # 其他定量生物学 (Other Quantitative Biology)
    "q-bio.PE",  # 群体与进化 (Populations and Evolution)
    "q-bio.QM",  # 定量方法 (Quantitative Methods)
    "q-bio.SC",  # 亚细胞过程 (Subcellular Processes)
    "q-bio.TO",  # 组织与器官 (Tissues and Organs)

    # 统计学 (Statistics)
    "stat.AP",  # 应用统计 (Applications)
    "stat.CO",  # 计算 (Computation)
    "stat.ML",  # 机器学习 (Machine Learning)
    "stat.ME",  # 方法论 (Methodology)
    "stat.OT",  # 其他统计 (Other Statistics)
    "stat.TH",  # 统计理论 (Statistics Theory)

    # 经济学 (Economics)
    "econ.EM",  # 计量经济学 (Econometrics)
    "econ.GN",  # 一般经济学 (General Economics)
    "econ.TH",  # 理论经济学 (Theoretical Economics)

    # 电气工程与系统科学 (Electrical Engineering and Systems Science)
    "eess.AS",  # 音频与语音处理 (Audio and Speech Processing)
    "eess.IV",  # 图像与视频处理 (Image and Video Processing)
    "eess.SP",  # 信号处理 (Signal Processing)
    "eess.SY",  # 系统与控制 (Systems and Control)

    # 量化金融 (Quantitative Finance)
    "q-fin.CP",  # 计算金融 (Computational Finance)
    "q-fin.EC",  # 经济学 (Economics)
    "q-fin.GN",  # 一般金融 (General Finance)
    "q-fin.MF",  # 数学金融 (Mathematical Finance)
    "q-fin.PM",  # 投资组合管理 (Portfolio Management)
    "q-fin.PR",  # 定价证券 (Pricing of Securities)
    "q-fin.RM",  # 风险管理 (Risk Management)
    "q-fin.ST",  # 统计金融 (Statistical Finance)
    "q-fin.TR",  # 交易与市场微观结构 (Trading and Market Microstructure)
]

def download_paper(result, category, output_dir):
    """
    下载论文并保存到指定目录

    Args:
        result: arXiv搜索结果
        category: 论文分类
        output_dir: 主输出目录

    Returns:
        bool: 下载是否成功
        str: 下载的文件路径
    """
    try:
        # 创建分类目录
        category_dir = output_dir / category.replace('.', '_')
        category_dir.mkdir(exist_ok=True)

        # 获取论文ID并处理可能的斜杠问题
        paper_id = result.get_short_id()
        safe_paper_id = paper_id.replace('/', '_')

        # 创建一个安全的文件名
        safe_title = "".join([c if c.isalnum() or c in [' ', '.', '-'] else '_' for c in result.title])
        safe_title = safe_title[:100]  # 限制文件名长度
        filename = f"{safe_paper_id}_{safe_title}.pdf"
        filepath = category_dir / filename

        # 如果文件已存在，跳过下载
        if filepath.exists():
            # 检查文件大小，确保不是损坏的文件
            if filepath.stat().st_size > 1000:  # 假设正常PDF至少有1KB
                return True, str(filepath)
            else:
                # 文件可能损坏，删除并重新下载
                filepath.unlink()

        # 下载PDF，最多尝试3次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 直接使用arxiv API下载PDF
                pdf_url = result.pdf_url
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status()  # 确保请求成功

                # 保存PDF文件
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # 验证文件是否成功下载
                if filepath.exists() and filepath.stat().st_size > 1000:
                    # 添加随机延迟以避免API限制
                    time.sleep(random.uniform(1, 3))
                    return True, str(filepath)
                else:
                    # 文件下载不完整，删除并重试
                    if filepath.exists():
                        filepath.unlink()

                    if attempt < max_retries - 1:
                        print(f"下载不完整，正在重试 ({attempt+1}/{max_retries}): {result.title}")
                        time.sleep(random.uniform(3, 5))  # 重试前等待更长时间
            except Exception as inner_e:
                if attempt < max_retries - 1:
                    print(f"下载出错，正在重试 ({attempt+1}/{max_retries}): {result.title}, 错误: {str(inner_e)}")
                    time.sleep(random.uniform(3, 5))
                else:
                    raise inner_e

        return False, None
    except Exception as e:
        print(f"下载失败: {result.title}, 错误: {str(e)}")
        return False, None

# 全局变量，用于处理中断
interrupted = False

def signal_handler(sig, frame):
    """处理中断信号"""
    global interrupted
    print("\n检测到中断信号，正在完成当前下载后退出...")
    interrupted = True

def load_progress():
    """加载下载进度"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载进度文件时出错: {str(e)}")

    # 如果文件不存在或加载失败，创建新的进度记录
    return {
        "completed": [],
        "failed": [],
        "categories": {}
    }

def save_progress(progress):
    """保存下载进度"""
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存进度文件时出错: {str(e)}")

def count_existing_papers():
    """计算已下载的论文数量"""
    # 递归查找所有PDF文件
    return len(list(OUTPUT_DIR.glob("**/*.pdf")))

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从arXiv下载论文")
    parser.add_argument("--total", type=int, default=1000, help="要下载的论文总数，默认为1000")
    return parser.parse_args()

def main():
    """主函数"""
    # 注册信号处理器，用于优雅地处理中断
    signal.signal(signal.SIGINT, signal_handler)

    # 解析命令行参数
    args = parse_arguments()
    total_papers = args.total

    # 加载进度
    progress = load_progress()

    # 统计已下载的论文数量
    existing_papers = count_existing_papers()
    completed_count = len(progress["completed"])

    # 如果已完成的数量与文件数不一致，可能是进度文件不完整
    if completed_count != existing_papers:
        print(f"警告：进度文件记录的完成数量 ({completed_count}) 与实际文件数量 ({existing_papers}) 不一致")

    # 计算已经下载的总数量
    total_downloaded_so_far = len(progress.get("completed", []))

    # 计算还需要下载的数量
    remaining_total = max(0, total_papers - total_downloaded_so_far)

    # 如果已经达到或超过目标数量，则直接结束
    if remaining_total <= 0:
        print(f"已达到目标下载数量：{total_downloaded_so_far}/{total_papers}")
        return

    # 计算每个分类需要下载的论文数量（确保每个分类至少有1篇，如果总数允许的话）
    # 首先检查是否有足够的论文分配给每个分类
    if remaining_total >= len(CATEGORIES):
        # 每个分类至少分配1篇
        min_per_category = 1
        remaining_after_min = remaining_total - len(CATEGORIES)
        extra_per_category = remaining_after_min // len(CATEGORIES)
        remaining_extras = remaining_after_min % len(CATEGORIES)

        # 创建分配计划
        category_allocation = {}
        for i, category in enumerate(CATEGORIES):
            # 已下载的数量
            already_downloaded = len(progress.get("categories", {}).get(category, []))
            # 基础分配：最小值 + 平均分配的额外数量 + 可能的额外1篇
            base_allocation = min_per_category + extra_per_category + (1 if i < remaining_extras else 0)
            # 最终分配：考虑已下载的数量
            category_allocation[category] = max(0, base_allocation - already_downloaded)
    else:
        # 如果总数不足以给每个分类分配1篇，则按优先级分配
        # 这里简单地按分类顺序分配，直到达到总数
        category_allocation = {}
        remaining = remaining_total
        for category in CATEGORIES:
            already_downloaded = len(progress.get("categories", {}).get(category, []))
            if remaining > 0:
                category_allocation[category] = 1
                remaining -= 1
            else:
                category_allocation[category] = 0

    downloaded_count = 0
    failed_count = 0

    print(f"计划从 {len(CATEGORIES)} 个不同领域下载共 {total_papers} 篇论文")
    print(f"已存在 {total_downloaded_so_far} 篇论文，还需下载 {remaining_total} 篇")

    # 第一轮：按计划下载
    for category, papers_to_download in category_allocation.items():
        if interrupted:
            break

        if papers_to_download <= 0:
            continue

        print(f"\n正在处理分类 {category}，计划下载 {papers_to_download} 篇")

        # 设置搜索参数
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=papers_to_download * 3,  # 获取更多结果以防下载失败
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        # 获取结果
        try:
            # 使用Client.results()方法替代已弃用的Search.results()
            results = list(client.results(search))

            if not results:
                print(f"分类 {category} 没有找到论文")
                continue

            # 使用tqdm显示进度条
            category_downloaded = 0
            for result in tqdm(results, desc=f"下载 {category} 论文", total=min(len(results), papers_to_download)):
                if interrupted or category_downloaded >= papers_to_download:
                    break

                # 检查是否已下载过该论文
                paper_id = result.get_short_id()
                if paper_id in progress.get("completed", []):
                    print(f"论文 {paper_id} 已下载，跳过")
                    continue

                # 下载论文
                success, filepath = download_paper(result, category, OUTPUT_DIR)

                if success:
                    # 更新进度
                    if category not in progress.get("categories", {}):
                        progress.setdefault("categories", {})[category] = []

                    # 添加到已完成列表
                    progress["completed"].append(paper_id)
                    progress["categories"].setdefault(category, []).append({
                        "id": paper_id,
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "published": result.published.strftime("%Y-%m-%d"),
                        "filepath": filepath
                    })

                    # 保存进度
                    save_progress(progress)

                    category_downloaded += 1
                    downloaded_count += 1
                else:
                    # 记录失败
                    progress.setdefault("failed", []).append({
                        "id": paper_id,
                        "title": result.title,
                        "category": category
                    })
                    save_progress(progress)

                    failed_count += 1

            print(f"分类 {category} 完成，本次成功下载 {category_downloaded} 篇论文")

        except Exception as e:
            print(f"处理分类 {category} 时出错: {str(e)}")

        # 添加延迟以避免API限制
        if not interrupted:
            time.sleep(random.uniform(2, 4))

    # 检查是否达到目标数量，如果没有，进行补充下载
    current_total = len(progress.get("completed", []))
    still_needed = max(0, total_papers - current_total)

    if still_needed > 0 and not interrupted:
        print(f"\n第一轮下载后还需 {still_needed} 篇论文，开始补充下载...")

        # 选择一些热门分类进行补充下载
        popular_categories = [
            "cs.AI", "cs.LG", "cs.CV", "math.ST", "stat.ML",
            "physics.comp-ph", "q-bio.BM", "econ.EM", "eess.SP",
            "astro-ph.CO", "cond-mat.mes-hall", "hep-th", "quant-ph",
            "math.NT", "math.AG", "physics.optics", "cs.DB", "cs.CR"
        ]

        # 如果热门分类不够，添加一些其他分类
        if len(popular_categories) < 20:
            for category in CATEGORIES:
                if category not in popular_categories:
                    popular_categories.append(category)
                    if len(popular_categories) >= 20:
                        break

        # 从热门分类中补充下载
        papers_per_popular = (still_needed + len(popular_categories) - 1) // len(popular_categories)

        # 记录已尝试过的论文ID，避免重复尝试
        attempted_paper_ids = set()

        # 尝试不同的排序方式
        sort_methods = [
            (arxiv.SortCriterion.SubmittedDate, arxiv.SortOrder.Descending),  # 最新的论文
            (arxiv.SortCriterion.Relevance, arxiv.SortOrder.Descending),      # 最相关的论文
            (arxiv.SortCriterion.SubmittedDate, arxiv.SortOrder.Ascending)    # 最早的论文
        ]

        # 尝试不同的搜索策略
        for sort_by, sort_order in sort_methods:
            if still_needed <= 0 or interrupted:
                break

            print(f"\n使用排序方式: {sort_by.name}, {sort_order.name}")

            for category in popular_categories:
                if interrupted or still_needed <= 0:
                    break

                print(f"\n补充下载：从分类 {category} 下载最多 {min(papers_per_popular, still_needed)} 篇论文")

                # 设置搜索参数 - 增加搜索结果数量
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=min(papers_per_popular, still_needed) * 10,  # 获取更多结果以提高找到未下载论文的概率
                    sort_by=sort_by,
                    sort_order=sort_order
                )

                try:
                    results = list(client.results(search))

                    if not results:
                        print(f"分类 {category} 没有找到更多论文")
                        continue

                    # 使用tqdm显示进度条
                    category_downloaded = 0
                    for result in tqdm(results, desc=f"补充下载 {category} 论文", total=min(len(results), still_needed)):
                        if interrupted or category_downloaded >= papers_per_popular or still_needed <= 0:
                            break

                        # 检查是否已下载过该论文或尝试过下载
                        paper_id = result.get_short_id()
                        if paper_id in progress.get("completed", []) or paper_id in attempted_paper_ids:
                            continue

                        # 记录已尝试过的论文ID
                        attempted_paper_ids.add(paper_id)

                        # 下载论文
                        success, filepath = download_paper(result, category, OUTPUT_DIR)

                        if success:
                            # 更新进度
                            if category not in progress.get("categories", {}):
                                progress.setdefault("categories", {})[category] = []

                            # 添加到已完成列表
                            progress["completed"].append(paper_id)
                            progress["categories"].setdefault(category, []).append({
                                "id": paper_id,
                                "title": result.title,
                                "authors": [author.name for author in result.authors],
                                "published": result.published.strftime("%Y-%m-%d"),
                                "filepath": filepath
                            })

                            # 保存进度
                            save_progress(progress)

                            category_downloaded += 1
                            downloaded_count += 1
                            still_needed -= 1
                        else:
                            # 记录失败
                            progress.setdefault("failed", []).append({
                                "id": paper_id,
                                "title": result.title,
                                "category": category
                            })
                            save_progress(progress)

                            failed_count += 1

                    print(f"补充下载分类 {category} 完成，本次成功下载 {category_downloaded} 篇论文")

                except Exception as e:
                    print(f"补充下载分类 {category} 时出错: {str(e)}")

                # 添加延迟以避免API限制
                if not interrupted:
                    time.sleep(random.uniform(2, 4))

        # 如果仍然没有达到目标，尝试使用更广泛的搜索
        if still_needed > 0 and not interrupted:
            print(f"\n仍需 {still_needed} 篇论文，尝试更广泛的搜索...")

            # 使用更广泛的搜索词
            search_terms = ["machine learning", "deep learning", "neural network",
                           "artificial intelligence", "computer vision", "natural language processing",
                           "quantum", "physics", "mathematics", "statistics", "biology", "economics"]

            for term in search_terms:
                if interrupted or still_needed <= 0:
                    break

                print(f"\n搜索关键词: {term}，计划下载 {still_needed} 篇论文")

                # 设置搜索参数
                search = arxiv.Search(
                    query=term,
                    max_results=still_needed * 5,
                    sort_by=arxiv.SortCriterion.Relevance,
                    sort_order=arxiv.SortOrder.Descending
                )

                try:
                    results = list(client.results(search))

                    if not results:
                        print(f"关键词 {term} 没有找到论文")
                        continue

                    # 使用tqdm显示进度条
                    term_downloaded = 0
                    for result in tqdm(results, desc=f"下载 {term} 论文", total=min(len(results), still_needed)):
                        if interrupted or term_downloaded >= still_needed:
                            break

                        # 检查是否已下载过该论文或尝试过下载
                        paper_id = result.get_short_id()
                        if paper_id in progress.get("completed", []) or paper_id in attempted_paper_ids:
                            continue

                        # 记录已尝试过的论文ID
                        attempted_paper_ids.add(paper_id)

                        # 获取论文的主分类
                        primary_category = result.primary_category

                        # 下载论文
                        success, filepath = download_paper(result, primary_category, OUTPUT_DIR)

                        if success:
                            # 更新进度
                            if primary_category not in progress.get("categories", {}):
                                progress.setdefault("categories", {})[primary_category] = []

                            # 添加到已完成列表
                            progress["completed"].append(paper_id)
                            progress["categories"].setdefault(primary_category, []).append({
                                "id": paper_id,
                                "title": result.title,
                                "authors": [author.name for author in result.authors],
                                "published": result.published.strftime("%Y-%m-%d"),
                                "filepath": filepath
                            })

                            # 保存进度
                            save_progress(progress)

                            term_downloaded += 1
                            downloaded_count += 1
                            still_needed -= 1
                        else:
                            # 记录失败
                            progress.setdefault("failed", []).append({
                                "id": paper_id,
                                "title": result.title,
                                "category": primary_category
                            })
                            save_progress(progress)

                            failed_count += 1

                    print(f"关键词 {term} 完成，本次成功下载 {term_downloaded} 篇论文")

                except Exception as e:
                    print(f"搜索关键词 {term} 时出错: {str(e)}")

                # 添加延迟以避免API限制
                if not interrupted:
                    time.sleep(random.uniform(2, 4))

    # 最终统计
    final_total = len(progress.get("completed", []))
    print(f"\n下载完成！本次下载 {downloaded_count} 篇论文，失败 {failed_count} 篇")
    print(f"总共有 {final_total} 篇论文在 {OUTPUT_DIR} 目录中")

    # 检查是否达到目标
    if final_total < total_papers:
        print(f"警告：未能达到目标下载数量 {total_papers}，实际下载 {final_total} 篇")
        print("可能的原因：")
        print("1. 某些分类没有足够的论文")
        print("2. API限制或网络问题导致下载失败")
        print("3. 部分论文已被删除或无法访问")
        print("建议：再次运行脚本尝试补充下载，或减少目标数量")

    if interrupted:
        print("下载被用户中断")
        sys.exit(1)

if __name__ == "__main__":
    main()
