# src/analysis/problem_analyzer.py
import pandas as pd
import numpy as np
import os
import re
from collections import Counter

try:
    from src.models.clustering_model import ProblemClusteringModel
    from src.models.topic_modeling import ProblemTopicModel
    from src.utils.text_processing import preprocess_text_pipeline
    from src.utils.feature_engineering_utils import parse_cost_value, parse_time_to_implement
except ImportError:
    import sys

    current_file_dir_analyzer = os.path.dirname(os.path.abspath(__file__))
    src_dir_analyzer = os.path.dirname(current_file_dir_analyzer)
    project_root_analyzer = os.path.dirname(src_dir_analyzer)
    if project_root_analyzer not in sys.path:
        sys.path.insert(0, project_root_analyzer)
    from src.models.clustering_model import ProblemClusteringModel
    from src.models.topic_modeling import ProblemTopicModel
    from src.utils.text_processing import preprocess_text_pipeline
    from src.utils.feature_engineering_utils import parse_cost_value, parse_time_to_implement

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
# *** المسار الصحيح للمحول الذي يستخدم التضمينات ***
CT_PREPROCESSOR_PATH_FOR_EMBEDDINGS = os.path.join(MODELS_DIR, 'ct_num_cat_embeddings_preprocessor.pkl')
BERTOPIC_MODEL_PATH = os.path.join(MODELS_DIR, 'bertopic_model.pkl')
FINAL_RESULTS_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'final_results_with_models.csv')


class ProblemAnalyzer:
    def __init__(self,
                 kmeans_path: str = KMEANS_MODEL_PATH,
                 ct_path: str = CT_PREPROCESSOR_PATH_FOR_EMBEDDINGS,  # *** استخدام المسار الصحيح ***
                 bertopic_path: str = BERTOPIC_MODEL_PATH,
                 profile_data_path: str = FINAL_RESULTS_DATA_PATH,
                 embedding_model_name_for_clustering: str = 'paraphrase-multilingual-MiniLM-L12-v2'
                 ):
        print("--- تهيئة ProblemAnalyzer ---")
        self.clustering_model = None
        self.topic_model = None
        self.df_profile_data = None

        try:
            print("تحميل نموذج التجميع (K-Means)...")
            temp_clustering_model = ProblemClusteringModel(
                kmeans_model_path=kmeans_path,
                ct_preprocessor_path=ct_path,
                embedding_model_name=embedding_model_name_for_clustering
            )
            if not all([temp_clustering_model.kmeans_model,
                        temp_clustering_model.column_transformer,
                        temp_clustering_model.sentence_model]):
                raise RuntimeError("فشل تحميل واحد أو أكثر من مكونات ProblemClusteringModel.")
            self.clustering_model = temp_clustering_model
            print("تم تحميل وتهيئة clustering_model بنجاح في ProblemAnalyzer.")
        except Exception as e_cluster:
            print(f"خطأ فادح أثناء تهيئة ProblemClusteringModel داخل ProblemAnalyzer: {e_cluster}")
            # لا تقم بتعيين self.clustering_model إذا حدث خطأ، سيبقى None

        try:
            print("\nتحميل نموذج تحليل الموضوعات (BERTopic)...")
            temp_topic_model = ProblemTopicModel(model_path=bertopic_path)
            if not temp_topic_model.model:
                raise RuntimeError("فشل تحميل نموذج BERTopic بشكل كامل.")
            self.topic_model = temp_topic_model
            print("تم تحميل وتهيئة topic_model بنجاح في ProblemAnalyzer.")
        except Exception as e_topic:
            print(f"خطأ فادح أثناء تهيئة ProblemTopicModel داخل ProblemAnalyzer: {e_topic}")

        try:
            if os.path.exists(profile_data_path):
                date_columns_to_parse_profiles = ['date_identified', 'date_closed', 'date_chosen',
                                                  'start_date_planned', 'end_date_planned',
                                                  'start_date_actual', 'end_date_actual']
                self.df_profile_data = pd.read_csv(profile_data_path, parse_dates=date_columns_to_parse_profiles)
                print(f"تم تحميل بيانات الملفات التعريفية من: {profile_data_path}")
            else:
                print(f"تحذير: ملف البيانات للملفات التعريفية '{profile_data_path}' غير موجود.")
        except Exception as e_profile:
            print(f"تحذير: خطأ أثناء تحميل بيانات الملفات التعريفية: {e_profile}")

        print("--- اكتملت تهيئة ProblemAnalyzer (مع التحقق من الأخطاء) ---")

    # ... (بقية دوال الكلاس: _prepare_input_data_for_clustering, _get_cluster_profile_summary,
    #      _get_topic_profile_summary, analyze_new_problem كما هي في الرد السابق الذي نجح معك) ...
    #      سأقوم بتضمينها كاملة للتأكيد
    def _prepare_input_data_for_clustering(self, problem_data: dict) -> pd.DataFrame:
        print("بدء _prepare_input_data_for_clustering (النسخة المحسنة)...")
        text_fields_to_combine = [
            problem_data.get('title', ''), problem_data.get('description_initial', ''),
            problem_data.get('refined_problem_statement_final', ''), problem_data.get('stakeholders_involved', ''),
            problem_data.get('initial_impact_assessment', ''), problem_data.get('problem_source', ''),
            problem_data.get('active_listening_notes', ''), problem_data.get('key_questions_asked', ''),
            problem_data.get('initial_hypotheses', ''), problem_data.get('key_findings_from_analysis', ''),
            problem_data.get('potential_root_causes_list', ''), problem_data.get('solution_description', ''),
            problem_data.get('justification_for_choice', ''), problem_data.get('what_went_well', ''),
            problem_data.get('what_could_be_improved', ''), problem_data.get('recommendations_for_future', ''),
            problem_data.get('key_takeaways', '')
        ]
        combined_raw_text = " ".join(
            filter(None, [str(t).strip() for t in text_fields_to_combine if pd.notna(t) and str(t).strip() != '']))
        processed_text_for_clustering = preprocess_text_pipeline(combined_raw_text)
        input_df_data = {}
        input_df_data[self.clustering_model.text_feature_col] = [processed_text_for_clustering]
        expected_numerical_features = self.clustering_model.numerical_features
        if 'estimated_cost_numeric' in expected_numerical_features:
            input_df_data['estimated_cost_numeric'] = [parse_cost_value(problem_data.get('estimated_cost'))]
        if 'overall_budget_numeric' in expected_numerical_features:
            input_df_data['overall_budget_numeric'] = [parse_cost_value(problem_data.get('overall_budget'))]
        if 'estimated_time_days' in expected_numerical_features:
            input_df_data['estimated_time_days'] = [
                parse_time_to_implement(problem_data.get('estimated_time_to_implement'))]
        if 'processed_text_length' in expected_numerical_features:
            input_df_data['processed_text_length'] = [len(processed_text_for_clustering.split())]
        for nf in expected_numerical_features:
            if nf not in input_df_data:
                input_df_data[nf] = [problem_data.get(nf, np.nan)]
                try:
                    input_df_data[nf] = [float(input_df_data[nf][0]) if pd.notna(input_df_data[nf][0]) else np.nan]
                except (ValueError, TypeError):
                    input_df_data[nf] = [np.nan]
        expected_categorical_features = self.clustering_model.categorical_features
        for cf in expected_categorical_features: input_df_data[cf] = [problem_data.get(cf)]
        df_for_prediction = pd.DataFrame(input_df_data)
        print("DataFrame قبل إرساله إلى clustering_model.predict (بعد التحويلات الأولية):")
        cols_to_print_debug = [col for col in (expected_numerical_features + expected_categorical_features + [
            self.clustering_model.text_feature_col]) if col in df_for_prediction.columns]
        if cols_to_print_debug:
            print(df_for_prediction[cols_to_print_debug].to_string())
        else:
            print("لا توجد أعمدة محددة للطباعة أو أن df_for_prediction فارغ.")
        return df_for_prediction

    def _get_cluster_profile_summary(self, cluster_id: int) -> str:
        if self.df_profile_data is None or 'cluster_kmeans' not in self.df_profile_data.columns:
            return "بيانات الملفات التعريفية للعناقيد غير متاحة."
        cluster_data = self.df_profile_data[self.df_profile_data['cluster_kmeans'] == cluster_id]
        if cluster_data.empty: return f"لا توجد مشاكل تاريخية معروفة تنتمي للعنقود K-Means رقم {cluster_id}."
        num_problems = len(cluster_data)
        summary_parts = [f"**العنقود {cluster_id}** (يضم **{num_problems}** مشكلة/مشاكل تاريخية مشابهة):"]
        numerical_profile = []
        if 'estimated_cost_numeric' in cluster_data and cluster_data['estimated_cost_numeric'].notna().any():
            avg_cost = cluster_data['estimated_cost_numeric'].mean()
            numerical_profile.append(f"متوسط التكلفة المقدرة ~ **{avg_cost:.2f}**")
        if 'estimated_time_days' in cluster_data and cluster_data['estimated_time_days'].notna().any():
            avg_time = cluster_data['estimated_time_days'].mean()
            numerical_profile.append(f"متوسط وقت التنفيذ المقدر ~ **{avg_time:.2f} يوم**")
        if numerical_profile: summary_parts.append("- " + "، ".join(numerical_profile) + ".")
        categorical_profile_parts = []
        categorical_cols_for_profile = ['domain', 'complexity_level', 'status', 'problem_source']
        for col in categorical_cols_for_profile:
            if col in cluster_data and cluster_data[col].notna().any():
                value_counts = cluster_data[col].value_counts(normalize=True)
                top_values = value_counts.head(2)
                col_profile_parts = []
                for val, प्रतिशत in top_values.items():
                    if val != 'Unknown' and प्रतिशत * 100 > 10:
                        col_profile_parts.append(f"{val} (بنسبة {प्रतिशत * 100:.0f}%)")
                if col_profile_parts: categorical_profile_parts.append(
                    f"{col.replace('_', ' ').capitalize()}: {', '.join(col_profile_parts)}")
        if categorical_profile_parts: summary_parts.append(
            "- الخصائص الفئوية الشائعة: " + "؛ ".join(categorical_profile_parts) + ".")
        if 'processed_text' in cluster_data:
            cluster_texts = cluster_data['processed_text'].dropna().loc[
                cluster_data['processed_text'].astype(str).str.strip() != '']
            if not cluster_texts.empty:
                full_cluster_text = " ".join(cluster_texts)
                words = full_cluster_text.split()
                if words:
                    top_keywords = [word for word, count in Counter(words).most_common(7)]
                    if top_keywords: summary_parts.append(
                        f"- أهم الكلمات المفتاحية في نصوص هذا العنقود: **{', '.join(top_keywords)}**.")
        if len(summary_parts) == 1: return summary_parts[
            0] + " لا توجد خصائص مميزة إضافية بارزة مسجلة لهذا العنقود حاليًا."
        return "\n".join(summary_parts)

    def _get_topic_profile_summary(self, topic_id: int) -> str:
        if self.topic_model is None or self.topic_model.model is None: return "نموذج تحليل الموضوعات غير محمل."
        try:
            num_problems_in_topic_str = "غير محدد"
            if self.df_profile_data is not None and 'bertopic_topic' in self.df_profile_data.columns:
                num_problems_in_topic = len(
                    self.df_profile_data[self.df_profile_data['bertopic_topic'] == int(topic_id)])
                num_problems_in_topic_str = str(num_problems_in_topic)
            if topic_id == -1: return f"المشكلة لم تتطابق مع موضوع محدد (صُنفت كموضوع ضوضاء/غير مميز، يضم {num_problems_in_topic_str} مشكلة تاريخية)."
            keywords_scores = self.topic_model.get_keywords_for_topic(topic_id)
            if not keywords_scores: return f"لا توجد كلمات رئيسية مميزة للموضوع رقم {topic_id} (يضم {num_problems_in_topic_str} مشكلة تاريخية)."
            keywords = [word for word, score in keywords_scores[:5]]
            keywords_str = "، ".join(keywords)
            topic_name_representation = f"موضوع {topic_id}"
            try:
                topic_info_single = self.topic_model.model.get_topic_info(topic_id)
                if topic_info_single is not None and not topic_info_single.empty:
                    topic_name_representation = topic_info_single['Name'].iloc[0].replace("_", " ").strip()
            except:
                pass
            return (f"**الموضوع {topic_id}** (الاسم التمثيلي: '{topic_name_representation}'):\n"
                    f"- يضم **{num_problems_in_topic_str}** مشكلة تاريخية مشابهة.\n"
                    f"- أهم الكلمات الدالة: **{keywords_str}**.")
        except Exception as e:
            return f"خطأ في استخلاص ملخص الموضوع {topic_id}: {e}"

    def analyze_new_problem(self, problem_data: dict) -> dict:
        analysis_results = {"input_problem_data": problem_data, "kmeans_cluster": None, "bertopic_topic": None,
                            "cluster_profile_summary": "لم يتم تحميل أو إنشاء ملف تعريف العنقود.",
                            "topic_profile_summary": "لم يتم تحميل أو إنشاء ملف تعريف الموضوع."}
        if not isinstance(problem_data, dict) or not problem_data:
            analysis_results["error"] = "بيانات المشكلة المدخلة غير صالحة."
            return analysis_results
        print(f"\n--- بدء تحليل مشكلة جديدة بعنوان: \"{problem_data.get('title', 'بدون عنوان')}\" ---")
        text_fields_for_topic = [problem_data.get('title', ''), problem_data.get('description_initial', ''),
                                 problem_data.get('refined_problem_statement_final', '')]
        combined_raw_text_for_topic = " ".join(
            filter(None, [str(t).strip() for t in text_fields_for_topic if pd.notna(t) and str(t).strip() != '']))
        cleaned_text_for_topic = preprocess_text_pipeline(combined_raw_text_for_topic)
        if self.topic_model and self.topic_model.model:
            if cleaned_text_for_topic.strip():
                topics, _ = self.topic_model.get_topics_for_texts([cleaned_text_for_topic])
                if topics is not None and len(topics) > 0:
                    analysis_results["bertopic_topic"] = topics[0]
                    print(f"موضوع BERTopic المتوقع: {topics[0]}")
                    analysis_results["topic_profile_summary"] = self._get_topic_profile_summary(topics[0])
                else:
                    print("BERTopic لم يتمكن من تحديد موضوع للنص.")
            else:
                print("النص المعالج لـ BERTopic فارغ، لا يمكن تحديد الموضوع.")
        else:
            print("نموذج BERTopic غير محمل، لا يمكن تحديد الموضوعات.")
        df_for_clustering = self._prepare_input_data_for_clustering(problem_data)
        if self.clustering_model and self.clustering_model.kmeans_model:
            if not df_for_clustering.empty and \
                    self.clustering_model.column_transformer and \
                    self.clustering_model.sentence_model:  # *** تحقق من sentence_model هنا ***
                cluster_prediction = self.clustering_model.predict(df_for_clustering)
                if cluster_prediction.size > 0:
                    analysis_results["kmeans_cluster"] = cluster_prediction[0]
                    print(f"عنقود K-Means المتوقع: {cluster_prediction[0]}")
                    analysis_results["cluster_profile_summary"] = self._get_cluster_profile_summary(
                        cluster_prediction[0])
                else:
                    print("K-Means لم يتمكن من التنبؤ بعنقود.")
            else:
                print("البيانات المدخلة لـ K-Means فارغة أو مكونات المعالجة/التضمين غير محملة.")
        else:
            print("نموذج K-Means غير محمل، لا يمكن التنبؤ بالعنقود.")
        print("--- اكتمل تحليل المشكلة ---")
        return analysis_results


# --- مثال للاستخدام (للاختبار) ---
if __name__ == '__main__':
    print("===== بدء اختبار ProblemAnalyzer =====")
    if not os.path.exists(FINAL_RESULTS_DATA_PATH):
        print(f"تحذير شديد: ملف البيانات للملفات التعريفية '{FINAL_RESULTS_DATA_PATH}' غير موجود!")
    analyzer = ProblemAnalyzer(profile_data_path=FINAL_RESULTS_DATA_PATH)
    if analyzer.clustering_model and analyzer.clustering_model.kmeans_model and analyzer.clustering_model.sentence_model and \
            analyzer.topic_model and analyzer.topic_model.model:  # تحقق شامل أكثر
        new_problem_1 = {
            'title': 'الشبكة بطيئة جدا في قسم المحاسبة',
            'description_initial': 'يشتكي الموظفون في قسم المحاسبة من بطء شديد في الوصول إلى الملفات. المشكلة بدأت منذ أسبوع.',
            'domain': 'تقني', 'complexity_level': 'متوسط', 'status': 'مفتوحة',
            'problem_source': 'شكاوى الموظفين', 'sentiment_label': 'سلبي',
            'estimated_cost': 'متوسط جدا', 'overall_budget': '5000-7000 دولار',
            'estimated_time_to_implement': 'حوالي 3 اسابيع'
        }
        new_problem_2 = {
            'title': 'طابعة لا تطبع',
            'description_initial': 'الطابعة لا تستجيب لأوامر الطباعة إطلاقا.',
            'domain': 'تقني', 'complexity_level': 'بسيط'
        }
        analysis_1 = analyzer.analyze_new_problem(new_problem_1)
        print("\n--- نتائج تحليل المشكلة 1 ---")
        for key, value in analysis_1.items(): print(f"  {key}: {value}")
        print("\n" + "=" * 30 + "\n")
        analysis_2 = analyzer.analyze_new_problem(new_problem_2)
        print("\n--- نتائج تحليل المشكلة 2 ---")
        for key, value in analysis_2.items(): print(f"  {key}: {value}")
    else:
        print("فشل تحميل أو تهيئة أحد النماذج أو مكوناته بشكل كامل. لا يمكن إجراء اختبار ProblemAnalyzer.")
    print("\n===== انتهاء اختبار ProblemAnalyzer =====")