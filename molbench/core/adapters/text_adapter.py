import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from molbench.core.adapters import BenchModel

class TextModel(BenchModel, ABC, nn.Module):
    """
    文字模型抽象基类
    输入：SMILES 字符串
    输出：预测值
    """
    _PARAM_SPECS = {
        'model_name': ('', str),
        'model_path': ('', str),  
        'num_classes': (1, int),
        'lr': (2e-5, float),
        'max_len': (128, int),
        'task_type': ('regression', str),
        'epochs': (3, int),
        'batch_size': (16, int),
        'weight_decay': (0.0, float),
        'random_state': (42, int),
        'pooling':('cls', str),
        'dropout':(0.1, float),
        'warmup_ratio':(0.1, float),
        'trust_remote_code':(False, bool),
    }

    def __init__(self, model_name: str = '', model_path: str = '', num_classes: int = 1,
                 lr: float=2e-5, max_len: int=128, task_type: str='regression', 
                 epochs: int=3, batch_size: int=16, **kw):
        nn.Module.__init__(self)

        self.model_name = model_name
        self.model_path = model_path  
        self.num_classes = num_classes
        self.lr = lr
        self.max_len = max_len
        self.task_type = task_type
        self.epochs = epochs
        self.batch_size = batch_size
        
        # 处理其他通过 kw 传递的参数
        for param, (default, typ) in self._PARAM_SPECS.items():
            if not hasattr(self, param):  # 如果还没设置
                val = kw.get(param, default)
                setattr(self, param, typ(val))
        
        # 保存剩余未识别的参数（向后兼容）
        self._extra_params = {k: v for k, v in kw.items() 
                             if k not in self._PARAM_SPECS}
        
        self._training_finished = False
        self._model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_scaler = None

        # 加载 tokenizer + 预训练模型
        self.tokenizer = None
        self.model = None
        self.loss_fn = None

    @abstractmethod
    def _init_model(self):
        """子类需要初始化 self.tokenizer 和 self.model"""
        pass
    def _init_loss(self):
        """根据任务类型初始化损失函数"""
        if self.task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        elif self.task_type == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else: # multiclass
            self.loss_fn = nn.CrossEntropyLoss()

    def finalize_init(self):
        """子类在设置完所有参数后调用"""
        if self._model_loaded and self.model is not None:
            return
        if self._training_finished and self.model is None:
            raise RuntimeError("模型已训练但 model 为 None， 实例可能被错误复制")
        if not self.model_path:
            raise ValueError("model_path 为空，无法初始化")
        
        self._init_model()
        self._init_loss()
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        self._model_loaded = True

    def _ensure_smiles_list(self, X) -> list[str]:
        """把任意输入转成字符串列表"""
        print(f"_ensure_smiles_list 输入: {type(X)}")
        
        if isinstance(X, pd.Series):
            result = X.astype(str).tolist()
            print(f"  pd.Series -> list, 长度 {len(result)}")
            return result
        
        if isinstance(X, np.ndarray):
            result = X.astype(str).tolist()
            print(f"  np.ndarray -> list, 长度 {len(result)}")
            return result
        
        if isinstance(X, list):
            result = [str(x) for x in X]
            print(f"  list -> list, 长度 {len(result)}")
            return result
        
        if isinstance(X, str):
            return [X]
        
        # 最后尝试
        try:
            result = [str(x) for x in X]
            print(f"  可迭代对象 -> list, 长度 {len(result)}")
            return result
        except:
            return [str(X)]

    def _tokenize(self, smiles_list: list[str]):
        """把 SMILES 列表转成模型输入的 token 字典"""
        return self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=self.max_len, return_tensors='pt')

    def fit(self, X: np.ndarray, y: np.ndarray):
        """统一训练流程"""
        self.finalize_init()
        X = self._ensure_smiles_list(X)
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        
        y = y.ravel()

        if self.task_type == 'regression' and self.label_scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.label_scaler = StandardScaler()
            y_scaled = self.label_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            print(f"  自动标签标准化: 原始 [{y.min():.2f}, {y.max():.2f}] -> "
                  f"标准化后 均值={y_scaled.mean():.4f}, 标准差={y_scaled.std():.4f}")
            print(f"  ✅ Scaler 已创建: mean={self.label_scaler.mean_[0]:.4f}, std={self.label_scaler.scale_[0]:.4f}")
        else:
            y_scaled = y

        # 1. 数据准备
        enc = self._tokenize(X)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float if self.task_type=='regression' else torch.long)
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(enc['input_ids'], enc['attention_mask'], y_tensor)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2. 训练循环
        self.train()
        if hasattr(self, 'get_optimizer_groups'):
            param_groups = self.get_optimizer_groups(self.lr)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr,
                weight_decay=self.weight_decay)
        
        # 学习率调度：预热 + 线性衰减
        total_steps = len(loader) * self.epochs
        warmup_steps = int(total_steps * getattr(self, 'warmup_ratio', 0.1))
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / 
                      float(max(1, total_steps - warmup_steps)))

        best_loss = float('inf') # 初始化最佳损失为无穷大
        patience = 3
        patience_counter = 0
        initial_lr = self.lr
        for epoch in range(self.epochs):  
            total_loss = 0               
            for input_ids, att, yb in loader:
                input_ids, att, yb = input_ids.to(self.device), att.to(self.device), yb.to(self.device)
                optimizer.zero_grad()

                outputs = self(input_ids, att, labels=yb)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # 兼容旧格式
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    if self.task_type in ('regression', 'binary'):
                        logits = logits.squeeze()
                    loss = self.loss_fn(logits, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss/len(loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # Loss 不下降，降低学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"  Loss 停滞，降低学习率至 {optimizer.param_groups[0]['lr']:.2e}")
                    patience_counter = 0
        
        self._training_finished = True
        self._model_loaded = True

        if hasattr(self, 'model') and hasattr(self.model, 'label_scaler'):
            self.model.label_scaler = self.label_scaler
            print(f"  ✅ 同步 scaler 到内部模型: mean={self.label_scaler.mean_[0]:.4f}")
        
        return self

    def predict(self, X):
        """
        统一的预测接口
        返回原始尺度的预测值
        """
        self.finalize_init()
        raise NotImplementedError("子类必须实现 predict 方法")

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """分类返回概率，回归返回 None"""
        if self.task_type == 'regression':
            return None
        
        self.finalize_init()
        self.model.eval()
        with torch.no_grad():
            enc = self._tokenize(X)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits

            if self.task_type == 'binary':
                probs = torch.sigmoid(logits).cpu().numpy()
                return np.column_stack([1 - probs, probs])  # 返回两列：负类概率和正类概率
            else:  # multiclass
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
    
    def get_params(self, deep=True):
        params = {}
        for param in self._PARAM_SPECS.keys():
            if hasattr(self, param):
                params[param] = getattr(self, param)
        params.update(self._extra_params)
        return params
    
    def set_params(self, **params):
        # 情况1：已训练完成
        if getattr(self, '_training_finished', False):
            # 只允许修改安全的超参数
            safe_keys = {'lr', 'batch_size', 'epochs', 'weight_decay', 'random_state'}
            safe_params = {k: v for k, v in params.items() if k in safe_keys}
            
            if safe_params:
                print(f"  [Trained] 更新安全参数: {list(safe_params.keys())}")
                for k, v in safe_params.items():
                    setattr(self, k, v)
            
            # 拒绝修改结构参数
            unsafe = set(params.keys()) - safe_keys
            if unsafe:
                print(f"  [Trained] 拒绝修改结构参数: {unsafe}")
            
            return self
        
        # 情况2：模型未训练
        old_model_path = str(getattr(self, 'model_path', ''))
        old_model_loaded = self._model_loaded

        for k, v in params.items():
            if k in self._PARAM_SPECS or hasattr(self, k):
                if k in self._PARAM_SPECS:
                    default, typ = self._PARAM_SPECS[k]
                    try:
                        v = typ(v)
                    except:
                        v = default
                setattr(self, k, v)
            else:
                self._extra_params[k] = v

        new_model_path = str(getattr(self, 'model_path', ''))
        
        if old_model_loaded and old_model_path != new_model_path:
            print(f"模型路径更新，重新加载: {new_model_path}")
            self._model_loaded = False

        return self

    def get_task_type(self):
        return self.task_type
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型到指定路径，子类需要实现"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """从指定路径加载模型，子类需要实现"""
        pass

class MolFormerForSequenceClassification(TextModel):
    """
    MolFormer 分类包装器
    """
    def __init__(self, base_model, num_labels, config, **kwargs):
        tokenizer = kwargs.pop('tokenizer', None)
        label_scaler = kwargs.pop('label_scaler', None)
        device = kwargs.pop('device', None)
        model_path = kwargs.pop('model_path', None)
        task_type = kwargs.pop('task_type', 'regression')
        max_len = kwargs.pop('max_len', 128)
        num_classes = kwargs.pop('num_classes', 1)
        
        super().__init__(**kwargs)
        # 显式设置保护参数
        self.tokenizer = tokenizer
        self.label_scaler = label_scaler
        if device is not None:
            self.device = device
        if model_path is not None:
            self.model_path = model_path
        self.task_type = task_type
        self.max_len = max_len
        self.num_classes = num_classes

        self.base_model = base_model
        self.num_labels = num_labels
        self.config = config
        
        # 分类头
        hidden_size = getattr(config, 'hidden_size', 768)
        self.dropout = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 初始化权重
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        if not hasattr(self, 'tokenizer') and 'tokenizer' in kwargs:
            self.tokenizer = kwargs['tokenizer']

    def _init_model(self):
        pass
    
    def get_optimizer_groups(self, base_lr):
        """返回优化器参数组，分类头用更高学习率"""
        # base_model 的参数
        base_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                base_params.append(param)
        
        return [
            {'params': base_params, 'lr': base_lr},  # base model 用基础学习率
            {'params': classifier_params, 'lr': base_lr * 10}  # 分类头用10倍
        ]
    
    def _prepare_labels(self, labels):
        """统一的标签预处理函数"""
        if labels is None:
            return None
            
        # 如果是 numpy 数组
        if isinstance(labels, np.ndarray):
            # 处理 object 类型
            if labels.dtype == np.object_:
                print(f"  ⚠️ 转换 object 类型标签为 float32")
                try:
                    labels = labels.astype(np.float32)
                except:
                    # 可能是字符串，用 LabelEncoder
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    labels = le.fit_transform(labels.astype(str)).astype(np.float32)
            
            # 处理 NaN
            if np.isnan(labels).any():
                print(f"  ⚠️ 发现 NaN 值，用 0 填充")
                labels = np.nan_to_num(labels, nan=0.0)
            
            # 转换为 tensor
            labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        
        # 如果是 tensor，确保是 float
        elif isinstance(labels, torch.Tensor):
            labels = labels.float().to(self.device)
        
        # 如果是列表或其他，尝试转换
        else:
            try:
                labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
            except:
                print(f"  ⚠️ 无法转换标签类型: {type(labels)}")
                return None
        
        return labels
    
    def fit(self, X, y):
        """重写 fit，复用父类逻辑"""
        X = self._ensure_smiles_list(X)
        return super().fit(X,y)
    
    def predict(self, X):
        """
        重写 predict：自动还原标准化
        """
        print(f"🔥 predict 中 label_scaler: {self.label_scaler is not None}")
        if self.label_scaler is not None:
            print(f"🔥 scaler mean: {self.label_scaler.mean_}")
            print(f"🔥 scaler scale: {self.label_scaler.scale_}")
        
        X = self._ensure_smiles_list(X)
        self.finalize_init()
        
        enc = self.tokenizer(X, padding=True, truncation=True, 
                            max_length=128, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        
        with torch.no_grad():
            outputs = self.forward(**enc)
            logits = outputs.logits
        
        pred_raw = logits.cpu().numpy()
        print(f"🔥 原始预测范围: [{pred_raw.min():.4f}, {pred_raw.max():.4f}]")
        print(f"🔥 原始预测均值: {pred_raw.mean():.4f}, 标准差: {pred_raw.std():.4f}")

        # 转换为 numpy
        if self.task_type == 'regression':

            pred_scaled = np.asarray(pred_raw).reshape(-1,1)
            
            if self.label_scaler is not None:
                if not hasattr(self.label_scaler, 'scale_'):
                    print("⚠️ Warning: label_scaler 未 fit，返回标准化值")
                    return pred_scaled.flatten()
                pred_2d = self.label_scaler.inverse_transform(pred_scaled)
                result = pred_2d.flatten()
                print(f"  预测值范围(反标准化后): [{result.min():.4f}, {result.max():.4f}]")
                return result
            else:
                return pred_scaled.flatten()
        else:
            return torch.softmax(logits, dim=-1).cpu().numpy()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        from transformers.modeling_outputs import SequenceClassifierOutput
        # 获取基础模型输出
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True
        }

        outputs = self.base_model(**model_kwargs)

        # 获取最后一层的 hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # hidden_states 是元组，最后一层是 [-1]
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Cannot get hidden states from model output")
        
        # 使用最后一个有效token的隐状态
        if attention_mask is not None:
            # 找到每个序列的最后一个非padding token
            last_token_indices = attention_mask.sum(dim=1) - 1  # [batch]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            # 提取最后一个token的隐状态
            pooled_output = hidden_states[batch_indices, last_token_indices]  # [batch, hidden]
        else:
            # 如果没有attention_mask，取序列最后一个位置
            pooled_output = hidden_states[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 计算 loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 回归
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.float())
            else:
                # 分类
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,  
            attentions=outputs.attentions,        
        )
    
    def save_pretrained(self, save_path):
        """保存"""
        self.base_model.save_pretrained(save_path)
        # 保存分类头
        torch.save({
            'classifier': self.classifier.state_dict(),
            'num_labels': self.num_labels,
            'config': self.config,
        }, f"{save_path}/classification_head.bin")
    
    def resize_token_embeddings(self, new_num_tokens):
        """调整词汇表"""
        self.base_model.resize_token_embeddings(new_num_tokens)
    
    def load(self, path):
        """
        实现抽象方法：加载模型
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # 加载 base_model
        self.base_model = AutoModelForCausalLM.from_pretrained(path)
        
        # 加载分类头
        head_path = f"{path}/classification_head.bin"
        if os.path.exists(head_path):
            head_data = torch.load(head_path, map_location=self.device)
            self.classifier.load_state_dict(head_data['classifier'])
            self.num_labels = head_data['num_labels']
            self.config = head_data['config']
        
        self.model = self.base_model  # TextModel 期望有 self.model
        self._model_loaded = True
        print(f"  ✓ MolFormer 模型已从 {path} 加载")
        return self
    
    def save(self, path):
        """
        实现抽象方法：保存模型
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # 保存 base_model
        self.base_model.save_pretrained(path)
        
        # 保存分类头
        torch.save({
            'classifier': self.classifier.state_dict(),
            'num_labels': self.num_labels,
            'config': self.config,
        }, f"{path}/classification_head.bin")
        
        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        
        print(f"  ✓ MolFormer 模型已保存到 {path}")

class HFTextModel(TextModel):
    """
    HuggingFace 预训练文本模型适配器
    支持 HuggingFace 模型（如 ChemBERTa2、Transformer 等）
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backend = 'huggingface'
    
    def _init_model(self):
        """初始化 HF 模型 - 统一处理本地和远程加载"""
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        from transformers import AutoModelForSequenceClassification
        
        if not self.model_path:
            raise ValueError("必须提供 model_path 或 model_name")
        
        print(f"加载模型：{self.model_path}")

        
        # 统一的加载函数
        def _load_model(local_only=True):
            load_kwargs = {
                'local_files_only': local_only,
                'trust_remote_code': getattr(self, 'trust_remote_code', False),
            }
            
            # 检查模型类型
            config = AutoConfig.from_pretrained(self.model_path, **load_kwargs)
            model_type = getattr(config, 'model_type', 'unknown')
            print(f"  模型类型: {model_type}")
            
            if self.model_name in ['molformer', 'GP-MoLFormer']:
                try:
                    tokenizer_path = getattr(self, 'tokenizer_path', 'ibm-research/MoLFormer-XL-both-10pct')
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        **load_kwargs,
                        use_fast=False
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    from transformers import AutoModelForCausalLM
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **load_kwargs
                    )

                    model_kwargs = {
                        'model_path': self.model_path,
                        'tokenizer': tokenizer,
                        'task_type': self.task_type,
                        'device': self.device,
                        'max_len': self.max_len,
                        'num_classes': self.num_classes,
                        'lr': self.lr,
                        'weight_decay': self.weight_decay,
                        'warmup_ratio': self.warmup_ratio,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size
                    }
                    model = MolFormerForSequenceClassification(
                        base_model=base_model,
                        num_labels=self.num_classes,
                        config=config,
                        **model_kwargs
                    )
                    model.label_scaler = self.label_scaler
                    model._model_loaded = self._model_loaded
                    model._training_finished = self._training_finished

                    self.model = model
                    print(f" ✅ GP-MoLFormer 实例创建完成")
                except Exception as e:
                    print(f"  ✗ GP-MoLFormer 加载失败: {e}")
                    raise
            else:
                # 加载 tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        **load_kwargs
                    )
                    print(f"  ✓ Fast Tokenizer {'从本地加载' if local_only else '从远程下载'}")
                except Exception as e:
                    print(f"  ✗ Fast tokenizer 加载失败: {str(e)[:80]}")
                    print(f"  尝试 slow tokenizer...")
                    load_kwargs['use_fast'] = False
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        **load_kwargs
                    )
                    print(f"  ✓ Slow tokenizer 加载成功")

                # 标准模型
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                    **load_kwargs
                )

                # 统一处理 padding token
                added_pad = False
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                        print(f"  设置 pad_token = eos_token ({tokenizer.eos_token})")
                    else:
                        tokenizer.add_special_tokens({'pad_token': ''})
                        added_pad = True
                        print("  添加新 pad_token")

                # 调整词汇表
                if added_pad or len(tokenizer) != model.config.vocab_size:
                    old_size = model.config.vocab_size
                    new_size = len(tokenizer)
                    print(f"  调整词汇表: {old_size} -> {new_size}")
                    model.resize_token_embeddings(new_size)
            
            print(f"  ✓ 模型加载完成 (词汇表: {len(tokenizer)})")
            return tokenizer, model
        
        # 尝试本地加载
        try:
            self.tokenizer, self.model = _load_model(local_only=True)
        except Exception as e:
            print(f"  ✗ 本地加载失败: {e}")
            print(f"  尝试远程下载...")
            
            # 远程下载使用相同逻辑
            try:
                self.tokenizer, self.model = _load_model(local_only=False)
            except Exception as e2:
                raise RuntimeError(f"无法加载模型 {self.model_path}: {e2}")
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """统一预测流程"""
        X = self._ensure_smiles_list(X)

        # 优先使用内部 model 自带的 predict
        if hasattr(self.model, 'predict') and callable(self.model.predict):
            print("🔥 使用内部 model 的 predict")
            self.model.label_scaler = self.label_scaler
            return self.model.predict(X)
        
        # 否则使用原有的预测逻辑
        print("🔥 使用 HFTextModel 的 predict")
        self.model.eval()
        with torch.no_grad():
            enc = self._tokenize(X)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
        
            # 兼容不同输出格式
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # 转换为 numpy
            if isinstance(logits, torch.Tensor):
                pred = logits.cpu().numpy()
            else:
                pred = np.array(logits)
            
            # 统一形状处理
            if pred.ndim > 2:
                pred = pred.reshape(pred.shape[0], -1)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.flatten()
            
            # 回归任务：反标准化
            if self.task_type == 'regression':
                # 确保 2D 用于 inverse_transform
                pred_2d = pred.reshape(-1, 1) if pred.ndim == 1 else pred
                
                # 关键：反标准化
                if self.label_scaler is not None:
                    pred_orig = self.label_scaler.inverse_transform(pred_2d).flatten()
                    print(f"  反标准化: 标准化[{pred.min():.3f},{pred.max():.3f}] → 原始[{pred_orig.min():.3f},{pred_orig.max():.3f}]")
                    return pred_orig
                else:
                    return pred_2d.flatten()
            
            elif self.task_type == 'binary':
                return (torch.sigmoid(torch.tensor(pred)) >= 0.5).int().numpy()
            
            else:  # multiclass
                return pred.argmax(axis=-1)   
            
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        统一的 forward 接口
        """
        # 调用底层模型
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        
        # 如果有 labels，计算 loss
        if labels is not None:
            logits = outputs.logits
            if self.task_type in ('regression', 'binary'):
                logits = logits.squeeze()
            loss = self.loss_fn(logits, labels)
            
            # 返回带 loss 的输出对象
            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(
                loss=loss,
                logits=outputs.logits,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            )
        
        return outputs
        
    def save(self, path: str) -> None:
        """保存模型和 tokenizer"""
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"HF 模型已保存到 {path}")
    
    def load(self, path: str) :
        """从路径加载模型和 tokenizer"""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        print(f"HF 模型已从 {path} 加载")
        return self

class DeepChemTextCNN(TextModel):
    """
    DeepChem TextCNN 模型适配器
    使用 CNN 处理 SMILES 字符串，非 Transformer 架构
    """
    _PARAM_SPECS = {
        'n_tasks': (1, int),
        'n_embedding': (128, int),
        'kernel_sizes': ([3, 4, 5], list),
        'filter_sizes': ([100, 200, 300], list),
        'dropout': (0.5, float),
    }
    def __init__(self, model_name: str = 'TextCNN', num_classes: int = 1,
                 n_embedding: int = 128, kernel_sizes: list = None, 
                 filter_sizes: list = None, dropout: float = 0.5, **kw):
        super().__init__(model_name=model_name, num_classes=num_classes,**kw)

        self.n_embedding = n_embedding
        self.kernel_sizes = kernel_sizes or [3,4,5]
        self.filter_sizes = filter_sizes or [100,200,300]
        self.dropout = dropout
        self.backend = 'deepchem'
        self.n_tasks = self.num_classes

        self._init_model()
        self._init_loss()
    
    def _init_model(self):
        """初始化 DeepChem TextCNNModel"""
        try:
            import deepchem as dc
        except ImportError:
            raise ImportError("需要安装 deepchem: pip install deepchem")
        from deepchem.models.text_cnn import default_dict

        print(f"初始化 TextCNN: n_tasks={self.num_classes}, "
            f"n_embedding={self.n_embedding}, seq_length={self.max_len}")
        self.model = dc.models.TextCNNModel(
            n_tasks=self.n_tasks,
            n_embedding=self.n_embedding,
            kernal_sizes=self.kernel_sizes,
            filter_sizes=self.filter_sizes,
            dropout=self.dropout,
            mode=self.task_type,
            char_dict= default_dict,      
            seq_length=self.max_len, 
        )
        
        # DeepChem 不需要外部 tokenizer，标记为 None, 实际 tokenize 在 model 内部处理
        self.tokenizer = None
    
    def _tokenize(self, smiles_list: list[str]):
        """DeepChem 不需要预 tokenize，直接返回原始数据"""
        return {'smiles': smiles_list}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """覆盖：使用 DeepChem 的数据流"""
        X = self._ensure_smiles_list(X)
        
        from deepchem.data import NumpyDataset
        dataset = NumpyDataset(X=np.array(X), y=y.reshape(-1, self.n_tasks))
        
        # DeepChem 自己的训练循环
        self.model.fit(dataset, nb_epoch=self.epochs)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._ensure_smiles_list(X)
        
        from deepchem.data import NumpyDataset
        dataset = NumpyDataset(X=np.array(X))
        
        preds = self.model.predict(dataset)
        
        if self.task_type == 'regression':
            return preds.squeeze()
        elif self.task_type == 'binary':
            return (preds >= 0.5).astype(int).squeeze()
        else:
            return preds.argmax(axis=1)
    
    def predict_proba(self, X: np.ndarray):
        if self.task_type == 'regression':
            return None
        
        X = self._ensure_smiles_list(X)
        from deepchem.data import NumpyDataset
        dataset = NumpyDataset(X=np.array(X))
        
        # DeepChem 可能直接返回概率或需要 softmax
        preds = self.model.predict(dataset)
        
        if self.task_type == 'binary':
            probs = torch.sigmoid(torch.tensor(preds)).numpy()
            return np.column_stack([1-probs, probs])
        else:
            return torch.softmax(torch.tensor(preds), dim=1).numpy()
    
    def get_params(self, deep=True):
        """合并父类参数和本类特有参数"""
        params = super().get_params(deep=deep)
        params.update({
            'n_embedding': self.n_embedding,
            'kernel_sizes': self.kernel_sizes,
            'filter_sizes': self.filter_sizes,
            'dropout': self.dropout,
        })
        return params
    
    def set_params(self, **params):
        """处理本类特有参数，其余传给父类"""
        own_params = ['n_embedding', 'kernel_sizes', 'filter_sizes', 'dropout']
        own_values = {}
        
        for k, v in params.items():
            if k in own_params:
                own_values[k] = v
            else:
                # 收集父类参数
                pass
        
        # 更新本类参数
        for k, v in own_values.items():
            setattr(self, k, v)
        
        # 调用父类处理其余参数
        super().set_params(**{k: v for k, v in params.items() if k not in own_params})
        
        # 模型相关参数改变时，需要重建
        if any(k in own_params for k in params.keys()):
            print(f"DeepChem TextCNN 参数已更新，重新初始化模型...")
            self._init_model()
        
        return self

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save_checkpoint(path)
        print(f"DeepChem TextCNN 已保存到 {path}")
    
    def load(self, path: str) -> "DeepChemTextCNN":
        self.model.restore(path)
        print(f"DeepChem TextCNN 已从 {path} 加载")
        return self


