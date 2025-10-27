# create_project_structure.py
import os


def create_project_structure():
    """创建完整的项目目录结构"""

    # 项目根目录
    root_dir = "breast_cancer_research"

    # 定义目录结构
    directories = [
        root_dir,
        os.path.join(root_dir, "data"),
        os.path.join(root_dir, "models"),
        os.path.join(root_dir, "visualization"),
        os.path.join(root_dir, "utils"),
        os.path.join(root_dir, "experiments"),
    ]

    # 创建所有目录
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[OK] 创建目录: {directory}")
        else:
            print(f"[EXISTS] 目录已存在: {directory}")

    # 定义所有文件
    files = {
        "data": ["WBCD.csv", "WDBC.csv"],
        "models": [
            "__init__.py",
            "feature_optimizer.py",
            "woa_optimizer.py",
            "stacking_ensemble.py",
            "model_evaluator.py"
        ],
        "visualization": [
            "__init__.py",
            "feature_analysis.py",
            "performance_plots.py"
        ],
        "utils": [
            "__init__.py",
            "data_loader.py",
            "metrics.py",
            "config.py"
        ],
        "experiments": [
            "wbcd_experiment.py",
            "wdbc_experiment.py"
        ],
        root_dir: [
            "main.py",
            "requirements.txt",
            "README.md"
        ]
    }

    # 创建所有文件
    for directory, file_list in files.items():
        for filename in file_list:
            file_path = os.path.join(directory, filename)
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 添加基础文件头
                    if filename.endswith('.py'):
                        f.write(f'"""{filename} - 模块描述"""\n\n')
                        if filename != '__init__.py':
                            f.write('# 待实现: 添加具体功能代码\n')
                print(f"[OK] 创建文件: {file_path}")
            else:
                print(f"[EXISTS] 文件已存在: {file_path}")

    print("\n[DONE] 项目结构创建完成！")
    print("[NEXT] 接下来我们将逐个文件进行内容填充")


