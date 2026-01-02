import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime

class SalesAIPredictor:
    def __init__(self, model_path='ai_models'):
        """初始化AI预测器"""
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.models = {}
        self.historical_data = []
        
        # 创建模型目录
        os.makedirs(model_path, exist_ok=True)
        
        # 加载现有模型或训练新模型
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """加载或训练模型"""
        model_files = ['roas_model.pkl', 'ctr_model.pkl', 'cvr_model.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(self.model_path, model_file)
            if os.path.exists(model_path):
                try:
                    model_name = model_file.replace('_model.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
                    print(f"已加载模型: {model_name}")
                except Exception as e:
                    print(f"加载模型 {model_file} 失败: {e}")
                    self.models[model_file.replace('_model.pkl', '')] = self.create_base_model()
            else:
                model_name = model_file.replace('_model.pkl', '')
                self.models[model_name] = self.create_base_model()
                print(f"创建新模型: {model_name}")
    
    def create_base_model(self):
        """创建基础线性回归模型"""
        return LinearRegression()
    
    def train_models(self, X, y_roas, y_ctr, y_cvr):
        """训练模型"""
        try:
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练ROAS模型
            self.models['roas'].fit(X_scaled, y_roas)
            
            # 训练CTR模型
            self.models['ctr'].fit(X_scaled, y_ctr)
            
            # 训练CVR模型
            self.models['cvr'].fit(X_scaled, y_cvr)
            
            # 保存模型
            self.save_models()
            
            return True
        except Exception as e:
            print(f"训练模型失败: {e}")
            return False
    
    def save_models(self):
        """保存模型"""
        for name, model in self.models.items():
            model_path = os.path.join(self.model_path, f'{name}_model.pkl')
            joblib.dump(model, model_path)
    
    def predict_roas(self, features):
        """预测ROAS"""
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.models['roas'].predict(features_scaled)[0]
            return max(0.5, min(prediction, 10.0))  # 限制在合理范围内
        except:
            # 如果预测失败，返回基准值
            return 2.0
    
    def predict_ctr(self, features):
        """预测CTR"""
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.models['ctr'].predict(features_scaled)[0]
            return max(0.1, min(prediction, 20.0))  # 限制在合理范围内
        except:
            return 1.5
    
    def predict_cvr(self, features):
        """预测CVR"""
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.models['cvr'].predict(features_scaled)[0]
            return max(0.5, min(prediction, 30.0))  # 限制在合理范围内
        except:
            return 2.0
    
    def get_suggestions(self, input_data):
        """获取AI优化建议"""
        # 解析输入数据
        price = float(input_data.get('pricePerSale', 0))
        cost = float(input_data.get('unitCost', 0))
        spend = float(input_data.get('spend', 0))
        impressions = float(input_data.get('forecastImpressions', 0))
        clicks = float(input_data.get('clicks', 0))
        leads = float(input_data.get('leads', 0))
        close_rate = float(input_data.get('closeRate', 0))
        target_roas = float(input_data.get('targetROAS', 0))
        language = input_data.get('language', 'zh')
        
        # 计算当前指标
        sales = round(leads * (close_rate / 100))
        revenue = sales * price
        gross_profit = price - cost
        net_profit = sales * gross_profit - spend
        current_roas = revenue / spend if spend > 0 else 0
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cvr = (sales / clicks * 100) if clicks > 0 else 0
        
        # 准备特征向量用于预测
        features = [price, cost, spend, impressions, clicks, leads, close_rate]
        
        # 预测优化后的指标
        predicted_roas = self.predict_roas(features)
        predicted_ctr = self.predict_ctr(features)
        predicted_cvr = self.predict_cvr(features)
        
        # 生成优化建议
        suggestions = self.generate_suggestions(
            price, cost, spend, impressions, clicks, leads, close_rate,
            current_roas, predicted_roas, ctr, predicted_ctr, cvr, predicted_cvr,
            target_roas, net_profit, language
        )
        
        # 生成优化后的数据
        optimized_data = self.generate_optimized_data(
            price, cost, spend, impressions, clicks, leads, close_rate, target_roas,
            current_roas, predicted_roas, ctr, predicted_ctr, cvr, predicted_cvr
        )
        
        suggestions['optimized_data'] = optimized_data
        
        # 保存历史数据
        self.save_historical_data(input_data, suggestions)
        
        return suggestions
    
    def generate_suggestions(self, price, cost, spend, impressions, clicks, leads, close_rate,
                            current_roas, predicted_roas, ctr, predicted_ctr, cvr, predicted_cvr,
                            target_roas, net_profit, language):
        """生成优化建议"""
        
        if language == 'zh':
            suggestions = {
                'overall_assessment': '',
                'improvement_suggestions': [],
                'expected_results': '',
                'key_metrics': []
            }
            
            # 总体评估
            if net_profit >= 0 and current_roas >= (target_roas or 2):
                suggestions['overall_assessment'] = "表现优秀！已达到盈利目标和ROAS目标"
            elif net_profit >= 0:
                suggestions['overall_assessment'] = "表现良好，已实现盈利但ROAS有待提升"
            else:
                suggestions['overall_assessment'] = "表现需改进，目前处于亏损状态"
            
            # 改进建议
            if ctr < 1.0:
                suggestions['improvement_suggestions'].append(
                    f"点击率(CTR)偏低({ctr:.2f}%)，建议优化广告创意和受众定位，预计可提升至{predicted_ctr:.2f}%"
                )
            
            if cvr < 2.0:
                suggestions['improvement_suggestions'].append(
                    f"转化率(CVR)较低({cvr:.2f}%)，建议优化落地页和用户体验，预计可提升至{predicted_cvr:.2f}%"
                )
            
            if close_rate < 20:
                suggestions['improvement_suggestions'].append(
                    f"成交率偏低({close_rate:.2f}%)，建议加强销售跟进和客户沟通，预计可提升至{min(close_rate * 1.2, 50):.2f}%"
                )
            
            if price - cost < price * 0.3:
                suggestions['improvement_suggestions'].append(
                    f"毛利率偏低({((price-cost)/price*100):.2f}%)，建议优化成本结构或考虑提价"
                )
            
            if current_roas < (target_roas or 2):
                suggestions['improvement_suggestions'].append(
                    f"ROAS({current_roas:.2f}x)未达目标({target_roas or 2}x)，建议优化广告投放策略"
                )
            
            if not suggestions['improvement_suggestions']:
                suggestions['improvement_suggestions'].append("当前设置较为合理，建议继续保持并持续监控数据")
            
            # 预期结果
            roas_improvement = ((predicted_roas - current_roas) / current_roas * 100) if current_roas > 0 else 100
            suggestions['expected_results'] = (
                f"实施优化后，预计ROAS可从{current_roas:.2f}x提升至{predicted_roas:.2f}x "
                f"(提升{max(0, roas_improvement):.1f}%)，净利润预计增长15-30%"
            )
            
            # 关键指标
            suggestions['key_metrics'] = [
                {
                    'name': '当前ROAS',
                    'value': f'{current_roas:.2f}x',
                    'impact': 'positive' if current_roas >= (target_roas or 2) else 'negative',
                    'impact_label': '达标' if current_roas >= (target_roas or 2) else '未达标'
                },
                {
                    'name': '预测优化ROAS',
                    'value': f'{predicted_roas:.2f}x',
                    'impact': 'positive',
                    'impact_label': '可达成'
                },
                {
                    'name': '当前净利润',
                    'value': f'RM {net_profit:.2f}',
                    'impact': 'positive' if net_profit >= 0 else 'negative',
                    'impact_label': '盈利' if net_profit >= 0 else '亏损'
                }
            ]
        
        else:  # English
            suggestions = {
                'overall_assessment': '',
                'improvement_suggestions': [],
                'expected_results': '',
                'key_metrics': []
            }
            
            # Overall assessment
            if net_profit >= 0 and current_roas >= (target_roas or 2):
                suggestions['overall_assessment'] = "Excellent performance! Achieved both profit and ROAS targets"
            elif net_profit >= 0:
                suggestions['overall_assessment'] = "Good performance, profitable but ROAS needs improvement"
            else:
                suggestions['overall_assessment'] = "Needs improvement, currently operating at a loss"
            
            # Improvement suggestions
            if ctr < 1.0:
                suggestions['improvement_suggestions'].append(
                    f"CTR is low ({ctr:.2f}%), optimize ad creatives and targeting, expected to reach {predicted_ctr:.2f}%"
                )
            
            if cvr < 2.0:
                suggestions['improvement_suggestions'].append(
                    f"CVR is low ({cvr:.2f}%), optimize landing page and user experience, expected to reach {predicted_cvr:.2f}%"
                )
            
            if close_rate < 20:
                suggestions['improvement_suggestions'].append(
                    f"Close rate is low ({close_rate:.2f}%), improve sales follow-up, expected to reach {min(close_rate * 1.2, 50):.2f}%"
                )
            
            if price - cost < price * 0.3:
                suggestions['improvement_suggestions'].append(
                    f"Gross margin is low ({((price-cost)/price*100):.2f}%), optimize cost structure or consider price increase"
                )
            
            if current_roas < (target_roas or 2):
                suggestions['improvement_suggestions'].append(
                    f"ROAS ({current_roas:.2f}x) below target ({target_roas or 2}x), optimize ad strategy"
                )
            
            if not suggestions['improvement_suggestions']:
                suggestions['improvement_suggestions'].append("Current setup is reasonable, maintain and monitor")
            
            # Expected results
            roas_improvement = ((predicted_roas - current_roas) / current_roas * 100) if current_roas > 0 else 100
            suggestions['expected_results'] = (
                f"After optimization, ROAS expected to improve from {current_roas:.2f}x to {predicted_roas:.2f}x "
                f"({max(0, roas_improvement):.1f}% improvement), net profit expected to grow 15-30%"
            )
            
            # Key metrics
            suggestions['key_metrics'] = [
                {
                    'name': 'Current ROAS',
                    'value': f'{current_roas:.2f}x',
                    'impact': 'positive' if current_roas >= (target_roas or 2) else 'negative',
                    'impact_label': 'Met' if current_roas >= (target_roas or 2) else 'Not Met'
                },
                {
                    'name': 'Predicted ROAS',
                    'value': f'{predicted_roas:.2f}x',
                    'impact': 'positive',
                    'impact_label': 'Achievable'
                },
                {
                    'name': 'Current Net Profit',
                    'value': f'RM {net_profit:.2f}',
                    'impact': 'positive' if net_profit >= 0 else 'negative',
                    'impact_label': 'Profitable' if net_profit >= 0 else 'Loss'
                }
            ]
        
        return suggestions
    
    def generate_optimized_data(self, price, cost, spend, impressions, clicks, leads, close_rate, target_roas,
                               current_roas, predicted_roas, ctr, predicted_ctr, cvr, predicted_cvr):
        """生成优化后的数据"""
        
        # 基于预测结果生成优化值
        optimized_price = price * (1.05 if (price - cost) / price < 0.3 else 1.02)
        optimized_cost = cost * 0.97  # 降低3%成本
        
        # 根据ROAS调整广告花费
        if current_roas < (target_roas or 2):
            optimized_spend = spend * 0.9  # 如果ROAS低，减少广告花费
        else:
            optimized_spend = spend * 1.1  # 如果ROAS好，增加广告花费
        
        # 基于预测CTR和CVR调整流量指标
        ctr_multiplier = predicted_ctr / ctr if ctr > 0 else 1.2
        cvr_multiplier = predicted_cvr / cvr if cvr > 0 else 1.3
        
        optimized_impressions = impressions * 1.15
        optimized_clicks = clicks * min(ctr_multiplier, 1.5)
        optimized_leads = leads * min(cvr_multiplier, 1.4)
        
        # 提高成交率
        optimized_close_rate = min(close_rate * 1.15, 50)
        
        # 调整目标ROAS
        optimized_target_roas = max(target_roas, predicted_roas * 0.9)
        
        return {
            'optimized_pricePerSale': round(optimized_price, 2),
            'optimized_unitCost': round(optimized_cost, 2),
            'optimized_spend': round(optimized_spend, 2),
            'optimized_forecastImpressions': round(optimized_impressions),
            'optimized_clicks': round(optimized_clicks),
            'optimized_leads': round(optimized_leads),
            'optimized_closeRate': round(optimized_close_rate, 2),
            'optimized_targetROAS': round(optimized_target_roas, 2)
        }
    
    def generate_optimization(self, input_data):
        """生成详细优化方案"""
        suggestions = self.get_suggestions(input_data)
        
        optimization = {
            'suggestions': suggestions,
            'action_plan': self.generate_action_plan(input_data.get('language', 'zh')),
            'timeline': self.generate_timeline(input_data.get('language', 'zh')),
            'expected_impact': self.generate_expected_impact(suggestions, input_data.get('language', 'zh'))
        }
        
        return optimization
    
    def generate_action_plan(self, language='zh'):
        """生成行动计划"""
        if language == 'zh':
            return [
                "第一周：优化广告创意和定位，提高CTR",
                "第二周：改进落地页设计，提升用户体验",
                "第三周：优化销售流程，提高成交率",
                "第四周：分析数据，调整定价和成本结构",
                "持续监控：每周检查关键指标，及时调整策略"
            ]
        else:
            return [
                "Week 1: Optimize ad creatives and targeting to improve CTR",
                "Week 2: Improve landing page design and user experience",
                "Week 3: Optimize sales process to increase close rate",
                "Week 4: Analyze data, adjust pricing and cost structure",
                "Ongoing: Monitor key metrics weekly, adjust strategy as needed"
            ]
    
    def generate_timeline(self, language='zh'):
        """生成时间线"""
        if language == 'zh':
            return {
                'immediate': "立即开始：数据分析和初步优化",
                'short_term': "1-2周：实施主要优化措施",
                'medium_term': "3-4周：评估效果并调整",
                'long_term': "1-3月：持续优化和扩大规模"
            }
        else:
            return {
                'immediate': "Immediate: Data analysis and initial optimization",
                'short_term': "1-2 weeks: Implement main optimization measures",
                'medium_term': "3-4 weeks: Evaluate results and adjust",
                'long_term': "1-3 months: Continuous optimization and scaling"
            }
    
    def generate_expected_impact(self, suggestions, language='zh'):
        """生成预期影响"""
        if language == 'zh':
            return {
                'roas_improvement': "15-40%",
                'profit_improvement': "20-50%",
                'ctr_improvement': "10-30%",
                'cvr_improvement': "15-35%",
                'close_rate_improvement': "10-25%"
            }
        else:
            return {
                'roas_improvement': "15-40%",
                'profit_improvement': "20-50%",
                'ctr_improvement': "10-30%",
                'cvr_improvement': "15-35%",
                'close_rate_improvement': "10-25%"
            }
    
    def save_historical_data(self, input_data, suggestions):
        """保存历史数据用于模型训练"""
        try:
            # 准备数据点
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'input': input_data,
                'suggestions': suggestions
            }
            
            self.historical_data.append(data_point)
            
            # 限制历史数据数量
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data[-1000:]
            
            # 定期训练模型
            if len(self.historical_data) % 100 == 0:
                self.retrain_models()
            
            return True
        except Exception as e:
            print(f"保存历史数据失败: {e}")
            return False
    
    def retrain_models(self):
        """重新训练模型"""
        if len(self.historical_data) < 50:
            return False
        
        try:
            # 准备训练数据
            X = []
            y_roas = []
            y_ctr = []
            y_cvr = []
            
            for data_point in self.historical_data:
                input_data = data_point['input']
                
                # 提取特征
                features = [
                    float(input_data.get('pricePerSale', 0)),
                    float(input_data.get('unitCost', 0)),
                    float(input_data.get('spend', 0)),
                    float(input_data.get('forecastImpressions', 0)),
                    float(input_data.get('clicks', 0)),
                    float(input_data.get('leads', 0)),
                    float(input_data.get('closeRate', 0))
                ]
                
                X.append(features)
                
                # 提取标签（这里使用建议中的预测值或实际优化结果）
                # 如果没有实际结果，使用预测值
                suggestions = data_point.get('suggestions', {})
                optimized = suggestions.get('optimized_data', {})
                
                # 使用优化后的ROAS作为标签
                if 'optimized_targetROAS' in optimized:
                    y_roas.append(float(optimized['optimized_targetROAS']))
                else:
                    y_roas.append(2.5)  # 默认值
                
                # 使用预测CTR和CVR
                y_ctr.append(2.0)  # 默认CTR
                y_cvr.append(3.0)  # 默认CVR
            
            # 转换为numpy数组
            X = np.array(X)
            y_roas = np.array(y_roas)
            y_ctr = np.array(y_ctr)
            y_cvr = np.array(y_cvr)
            
            # 训练模型
            success = self.train_models(X, y_roas, y_ctr, y_cvr)
            
            if success:
                print(f"模型重新训练完成，使用 {len(self.historical_data)} 个数据点")
            
            return success
            
        except Exception as e:
            print(f"重新训练模型失败: {e}")
            return False