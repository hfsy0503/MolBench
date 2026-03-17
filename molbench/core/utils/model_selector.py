import json
import os
from pathlib import Path
from skopt.space import Integer, Real, Categorical

class UnifiedModelSelector:
    def __init__(self, models_dir='model_configs', extra_dirs=None):
        self.models_dir = Path(models_dir)
        # 默认额外目录为 hyper_parameters/define，允许用户将自定义 json 放入该目录
        if extra_dirs is None:
            extra_dirs = ['hyper_parameters/define']
        self.extra_dirs = [Path(p) for p in extra_dirs]
        self.ensure_models_dir()
        self.ensure_extra_dirs()
    
    def ensure_models_dir(self):
        """确保模型配置目录存在"""
        if not self.models_dir.exists():
            self.models_dir.mkdir()
            self.create_example_configs()
    
    def get_available_models(self):
        """获取所有可用的模型名称"""
        names = set()
        # 主目录
        for file in self.models_dir.glob('*.json'):
            names.add(file.stem)

        # 额外目录（如 hyper_parameters/define）
        for d in self.extra_dirs:
            try:
                for file in d.glob('*.json'):
                    names.add(file.stem)
            except Exception:
                continue

        return sorted(names)
    
    def load_single_model(self, model_name):
        """加载单个模型配置，优先从 models_dir，然后在 extra_dirs 中查找"""
        search_paths = [self.models_dir] + self.extra_dirs
        for base in search_paths:
            file_path = base / f"{model_name}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        raise FileNotFoundError(f"模型配置文件不存在: {model_name}.json in {search_paths}")

    def ensure_extra_dirs(self):
        """确保所有额外目录存在"""
        for d in self.extra_dirs:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
    
    def load_models(self, selection=None):
        """
        统一的模型加载方法
        
        参数:
        - selection: 
            None 或 "all": 加载所有模型
            str: 单个模型名称，如 'RandomForest'
            list: 加载指定模型列表, 如 ['RandomForest', 'SVM'] 或 [{'name': 'RF', 'path': 'xxx.json'}]
            "interactive": 交互式选择
        """
        available_models = self.get_available_models()
        
        if not available_models:
            raise FileNotFoundError(f"在 {self.models_dir} 中没有找到任何模型配置")
        
        if selection is None or selection == "all":
            print("📁 加载所有可用模型...")
            return self._load_model_list(available_models)
        
        elif isinstance(selection, str) and selection not in ["interactive", "all"]:
            print("📁 加载指定模型...")
            if selection not in available_models:
                raise ValueError(f"模型 '{selection}' 不可用。可用模型: {available_models}")
            return self._load_model_list([selection])
        
        elif isinstance(selection, list):
            return self._load_model_list(selection)
        
        elif selection == "interactive":
            return self._interactive_selection(available_models)
        
        else:
            raise ValueError("无效的选择参数")
    
    def _load_model_list(self, model_list):
        """加载模型列表"""
        configs = {}
        for model_name in model_list:
            try:
                configs[model_name] = self.load_single_model(model_name)
                print(f"   ✅ {model_name}")
            except Exception as e:
                print(f"   ❌ {model_name}: {e}")
        
        print(f"\n🎯 成功加载 {len(configs)}/{len(model_list)} 个模型")
        return configs
    
    def _interactive_selection(self, available_models):
        """交互式选择模型"""
        print("\n" + "="*60)
        print("🤖 机器学习模型选择器")
        print("="*60)
        
        # 显示所有可用模型
        print("\n可用的模型:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")
        
        print("\n选择方式:")
        print("  1. 选择所有模型")
        print("  2. 手动选择特定模型")
        print("  3. 按类型筛选")
        
        while True:
            try:
                choice = input("\n请选择 (1/2/3): ").strip()
                
                if choice == '1':
                    return self._load_model_list(available_models)
                
                elif choice == '2':
                    return self._manual_selection(available_models)
                
                elif choice == '3':
                    return self._filter_selection(available_models)
                
                else:
                    print("❌ 无效选择，请重新输入")
            except (ValueError, IndexError):
                print("❌ 输入格式错误，请重新输入")
    
    def _manual_selection(self, available_models):
        """手动选择模型"""
        print("\n请输入模型编号(多个用逗号分隔，如: 1,3,5):")
        print("  或输入 'all' 选择所有模型")
        
        try:
            user_input = input("模型编号: ").strip()
            
            if user_input.lower() == 'all':
                return self._load_model_list(available_models)
            
            # 解析用户输入
            indices = [int(x.strip()) for x in user_input.split(',')]
            selected_models = []
            
            for idx in indices:
                if 1 <= idx <= len(available_models):
                    selected_models.append(available_models[idx-1])
                else:
                    print(f"⚠️  忽略无效编号: {idx}")
            
            if not selected_models:
                print("⚠️  未选择任何有效模型，使用所有模型")
                return self._load_model_list(available_models)
            
            return self._load_model_list(selected_models)
            
        except ValueError:
            print("❌ 输入格式错误，使用所有模型")
            return self._load_model_list(available_models)
    
    def _filter_selection(self, available_models):
        """按类型筛选模型"""
        print("\n筛选方式:")
        print("  1. 回归模型 (Regressor)")
        print("  2. 分类模型 (Classifier)") 
        print("  3. 树模型 (Tree, Forest)")
        print("  4. 线性模型 (Linear, Logistic)")
        print("  5. SVM模型")
        print("  6. 神经网络")
        
        try:
            filter_choice = input("请选择筛选类型 (1-6): ").strip()
            
            if filter_choice == '1':
                selected = [m for m in available_models if 'Regressor' in m]
            elif filter_choice == '2':
                selected = [m for m in available_models if 'Classifier' in m]
            elif filter_choice == '3':
                selected = [m for m in available_models if any(x in m for x in ['Tree', 'Forest'])]
            elif filter_choice == '4':
                selected = [m for m in available_models if any(x in m for x in ['Linear', 'Logistic'])]
            elif filter_choice == '5':
                selected = [m for m in available_models if 'SVC' in m or 'SVR' in m]
            elif filter_choice == '6':
                selected = [m for m in available_models if 'MLP' in m]
            else:
                print("❌ 无效选择，使用所有模型")
                selected = available_models
            
            if not selected:
                print("⚠️  没有找到匹配的模型，使用所有模型")
                selected = available_models
            
            return self._load_model_list(selected)
            
        except Exception:
            print("❌ 选择错误，使用所有模型")
            return self._load_model_list(available_models)
    
    def create_example_configs(self):
        """创建示例配置文件"""
        examples = {
            'RandomForestRegressor.json': {
                "model": "RandomForestRegressor",
                "params": {
                    "n_estimators": {"type": "integer", "bounds": [10, 500]},
                    "max_depth": {"type": "integer", "bounds": [1, 30]}
                },
                "fixed_params": {"random_state": 42}
            },
            'LogisticRegression.json': {
                "model": "LogisticRegression", 
                "params": {
                    "C": {"type": "real", "bounds": [0.01, 100], "prior": "log-uniform"}
                },
                "fixed_params": {"random_state": 42}
            }
        }
        
        for filename, config in examples.items():
            filepath = self.models_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"📝 创建示例配置: {filename}")