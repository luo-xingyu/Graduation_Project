"""
论文生成器命令行界面
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional
from paper_generator import PaperGenerator

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="自动生成学术论文工具")
    
    parser.add_argument("--topic", "-t", type=str, required=True,
                        help="论文主题")
    
    parser.add_argument("--output", "-o", type=str, default="generated_paper.md",
                        help="输出文件路径 (默认: generated_paper.md)")
    
    parser.add_argument("--format", "-f", type=str, choices=["markdown", "text"], default="markdown",
                        help="输出格式 (默认: markdown)")
    
    parser.add_argument("--citation-style", "-c", type=str, default="APA",
                        help="引用样式 (默认: APA)")
    
    parser.add_argument("--api-key", "-k", type=str,
                        help="OpenAI API密钥 (默认使用环境变量OPENAI_API_KEY)")
    
    parser.add_argument("--api-url", "-u", type=str,
                        help="API端点URL (默认使用环境变量OPENAI_API_URL或标准OpenAI URL)")
    
    parser.add_argument("--model", "-m", type=str, default="gpt-4-1106-preview",
                        help="使用的模型名称 (默认: gpt-4-1106-preview)")
    
    parser.add_argument("--sections", "-s", type=str, nargs="+",
                        choices=["Abstract", "Keywords", "Introduction", "Literature Review",
                                "Methodology", "Results", "Discussion", "Conclusion", "References", "all"],
                        default=["all"],
                        help="要生成的部分 (默认: all)")
    
    parser.add_argument("--humanize", action="store_true",
                        help="对生成的文本进行人性化处理，使其更像人类写作")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细输出")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置API密钥
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 未提供API密钥。请使用--api-key参数或设置OPENAI_API_KEY环境变量。")
        sys.exit(1)
    
    # 设置API URL
    api_url = args.api_url or os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
    
    # 创建生成器
    generator = PaperGenerator(api_key=api_key, api_url=api_url, api_model=args.model)
    
    # 确定要生成的部分
    if "all" in args.sections:
        include_sections = [
            "Abstract", "Keywords", "Introduction", "Literature Review",
            "Methodology", "Results", "Discussion", "Conclusion", "References"
        ]
    else:
        include_sections = args.sections
    
    # 显示配置信息
    if args.verbose:
        print("\n=== 配置信息 ===")
        print(f"主题: {args.topic}")
        print(f"输出文件: {args.output}")
        print(f"格式: {args.format}")
        print(f"引用样式: {args.citation_style}")
        print(f"模型: {args.model}")
        print(f"要生成的部分: {', '.join(include_sections)}")
        print(f"人性化处理: {'是' if args.humanize else '否'}")
        print(f"API URL: {api_url}")
        print("=" * 30 + "\n")
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 生成论文
        print(f"开始生成关于 '{args.topic}' 的论文...")
        generator.generate_full_paper(
            topic=args.topic,
            citation_style=args.citation_style,
            include_sections=include_sections
        )
        
        # 人性化处理
        if args.humanize:
            print("进行人性化处理...")
            generator.humanize_paper()
        
        # 保存到文件
        generator.save_paper_to_file(args.output, args.format)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        print(f"\n论文生成完成！耗时: {elapsed_time:.2f} 秒")
        print(f"输出文件: {os.path.abspath(args.output)}")
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        sys.exit(1)
    except Exception as e:
        print(f"\n生成过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
