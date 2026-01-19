#!/usr/bin/env python3
"""
临时后处理脚本：修复 entities.jsonl 中的重复实体记录

同一文档内的同名实体只保留第一条记录（prop_idx 最小的）
"""

import json
import argparse
from pathlib import Path
from collections import OrderedDict


def fix_duplicate_entities(input_file: str, output_file: str = None, dry_run: bool = False):
    """
    修复 entities.jsonl 中的重复实体记录

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为 None，覆盖原文件）
        dry_run: 是否只模拟运行，不实际写入文件
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_file}")
        return

    # 读取并去重
    print(f"读取文件: {input_file}")
    entities_seen = OrderedDict()  # 使用 OrderedDict 保持第一次出现的顺序
    duplicates_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entity = json.loads(line)
                entity_key = (entity.get('doc_id', ''), entity.get('text', ''))

                if entity_key not in entities_seen:
                    entities_seen[entity_key] = entity
                else:
                    duplicates_count += 1
                    # 可以打印重复信息（注释掉以减少输出）
                    # print(f"  发现重复: line={line_num}, doc_id={entity_key[0]}, text={entity_key[1]}")
            except json.JSONDecodeError as e:
                print(f"警告: line {line_num} JSON 解析失败: {e}")

    print(f"\n统计信息:")
    print(f"  原始记录数: {len(entities_seen) + duplicates_count}")
    print(f"  去重后记录数: {len(entities_seen)}")
    print(f"  过滤重复数: {duplicates_count}")

    # 确定输出文件路径
    if output_file is None:
        output_path = input_path  # 覆盖原文件
    else:
        output_path = Path(output_file)

    # 写入文件
    if dry_run:
        print(f"\n[DRY RUN] 不会写入文件: {output_path}")
    else:
        print(f"\n写入文件: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entity in entities_seen.values():
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")

        print(f"完成! 已修复 {duplicates_count} 条重复记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="修复 entities.jsonl 中的重复实体记录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 修复单个文件（覆盖原文件）
  python scripts/temp_fix_duplicate_entities.py \\
    --input output/HotpotQA/proposition_graph/meta/entities.jsonl

  # 修复并保存到新文件
  python scripts/temp_fix_duplicate_entities.py \\
    --input output/HotpotQA/proposition_graph/meta/entities.jsonl \\
    --output output/HotpotQA/proposition_graph/meta/entities_fixed.jsonl

  # 模拟运行（不实际写入）
  python scripts/temp_fix_duplicate_entities.py \\
    --input output/HotpotQA/proposition_graph/meta/entities.jsonl \\
    --dry-run

  # 批量处理多个数据集
  for dataset in HotpotQA 2WikiMultihopQA MuSiQue; do
    python scripts/temp_fix_duplicate_entities.py \\
      --input output/$dataset/proposition_graph/meta/entities.jsonl
  done
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文件路径 (entities.jsonl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（默认覆盖原文件）'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际写入文件'
    )

    args = parser.parse_args()

    fix_duplicate_entities(
        input_file=args.input,
        output_file=args.output,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
