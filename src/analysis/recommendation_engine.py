# src/analysis/recommendation_engine.py
import pandas as pd
import numpy as np
import os

# --- تعريف مسارات الملفات ---
# نفترض أن هذا الملف موجود في src/analysis/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# *** التصحيح: تعريف المتغير قبل استخدامه ***
# ملف البيانات الذي يحتوي على تعيينات العناقيد والموضوعات والمعلومات الأصلية
# هذا هو الملف الذي يفترض أننا حفظناه من 03_results_analysis.ipynb
HISTORICAL_DATA_WITH_ALL_RESULTS_PATH = os.path.join(PROCESSED_DATA_DIR, 'final_results_with_models.csv')


# هذا المسار لم يعد ضروريًا كقيمة افتراضية إذا استخدمنا المسار المدمج
# HISTORICAL_DATA_WITH_CLUSTERS_PATH = os.path.join(PROCESSED_DATA_DIR, 'problems_with_kmeans_clusters.csv')


class RecommendationEngine:
    # *** استخدام المتغير المعرف أعلاه كقيمة افتراضية ***
    def __init__(self, historical_data_path: str = HISTORICAL_DATA_WITH_ALL_RESULTS_PATH):
        print("--- تهيئة RecommendationEngine ---")
        self.historical_data = None
        try:
            date_columns_to_parse_rec = ['date_identified', 'date_closed', 'date_chosen',
                                         'start_date_planned', 'end_date_planned',
                                         'start_date_actual', 'end_date_actual']
            self.historical_data = pd.read_csv(historical_data_path, parse_dates=date_columns_to_parse_rec)
            print(f"تم تحميل البيانات التاريخية للتوصيات من: {historical_data_path}")
            print(f"أبعاد البيانات التاريخية: {self.historical_data.shape}")

            required_cols = ['problem_id', 'title',
                             'cluster_kmeans', 'bertopic_topic',
                             'solution_description', 'what_went_well',
                             'what_could_be_improved', 'recommendations_for_future']
            missing_cols = [col for col in required_cols if col not in self.historical_data.columns]
            if missing_cols:
                print(f"تحذير: الأعمدة التالية مفقودة من البيانات التاريخية وقد تؤثر على التوصيات: {missing_cols}")

        except FileNotFoundError:
            print(f"خطأ: ملف البيانات التاريخية '{historical_data_path}' غير موجود.")
            print(f"يرجى التأكد من إنشاء وحفظ ملف 'final_results_with_models.csv' في مجلد 'data/processed/'")
            print(f"من خلال تشغيل الخلية الأولى في دفتر '03_results_analysis.ipynb' بشكل صحيح.")
        except Exception as e:
            print(f"خطأ أثناء تحميل البيانات التاريخية: {e}")

        print("--- اكتملت تهيئة RecommendationEngine ---")

    def _extract_recommendations_from_df(self, df_similar: pd.DataFrame, top_n: int) -> list:
        """دالة مساعدة لاستخلاص وتنسيق التوصيات من DataFrame لمشاكل مشابهة."""
        recommendations = []
        if df_similar.empty:
            return recommendations

        if 'solution_description' in df_similar.columns:
            # التأكد من أن القيم نصية قبل تطبيق .str (لتجنب خطأ مع float NaN مثلاً)
            valid_solutions = df_similar['solution_description'].dropna().astype(str)
            past_solutions = valid_solutions.loc[valid_solutions.str.strip() != ''].unique()
            if len(past_solutions) > 0:
                recs = [f"الحل المقترح/المختار سابقًا: '{sol}'" for sol in past_solutions[:top_n]]
                recommendations.extend(recs)

        lessons_fields = {
            'what_went_well': "ما سار على ما يرام سابقًا",
            'what_could_be_improved': "ما كان يمكن تحسينه سابقًا",
            'recommendations_for_future': "توصيات للمستقبل من مشاكل مشابهة"
        }
        for field, desc in lessons_fields.items():
            if field in df_similar.columns:
                valid_lessons = df_similar[field].dropna().astype(str)
                lessons = valid_lessons.loc[valid_lessons.str.strip() != ''].unique()
                if len(lessons) > 0:
                    recs = [f"{desc}: '{lesson}'" for lesson in lessons[:top_n]]
                    recommendations.extend(recs)
        return recommendations

    def get_recommendations(self, problem_analysis_results: dict, top_n: int = 3) -> dict:
        recommendations_output = {
            "based_on_kmeans_cluster": [],
            "based_on_bertopic_topic": [],
            "general_warnings": []
        }

        if self.historical_data is None or self.historical_data.empty:
            recommendations_output["general_warnings"].append("لا توجد بيانات تاريخية متاحة لتقديم توصيات.")
            return recommendations_output

        # تأكد أن problem_analysis_results هو قاموس صالح
        if not isinstance(problem_analysis_results, dict):
            recommendations_output["general_warnings"].append("بيانات تحليل المشكلة المدخلة غير صالحة.")
            return recommendations_output

        current_problem_id = problem_analysis_results.get("input_problem_data", {}).get("problem_id")

        # --- 1. توصيات بناءً على عنقود K-Means ---
        kmeans_cluster = problem_analysis_results.get("kmeans_cluster")
        if kmeans_cluster is not None and 'cluster_kmeans' in self.historical_data.columns:
            print(f"\nالبحث عن توصيات بناءً على عنقود K-Means رقم: {kmeans_cluster}")
            # التأكد أن current_problem_id ليس None قبل المقارنة
            if current_problem_id is not None:
                similar_problems_k = self.historical_data[
                    (self.historical_data['cluster_kmeans'] == kmeans_cluster) &
                    (self.historical_data['problem_id'] != current_problem_id)
                    ]
            else:  # إذا لم يكن للمشكلة الجديدة ID، قارن بكل المشاكل في العنقود
                similar_problems_k = self.historical_data[
                    (self.historical_data['cluster_kmeans'] == kmeans_cluster)
                ]

            if not similar_problems_k.empty:
                print(f"تم العثور على {len(similar_problems_k)} مشكلة مشابهة في نفس عنقود K-Means.")
                recommendations_output["based_on_kmeans_cluster"] = self._extract_recommendations_from_df(
                    similar_problems_k, top_n)
            else:
                recommendations_output["general_warnings"].append(
                    f"لم يتم العثور على مشاكل أخرى في عنقود K-Means رقم {kmeans_cluster} (باستثناء المشكلة الحالية إذا كان لها ID).")
        elif kmeans_cluster is None:
            recommendations_output["general_warnings"].append("لم يتم تحديد عنقود K-Means للمشكلة الحالية.")
        elif 'cluster_kmeans' not in self.historical_data.columns:
            recommendations_output["general_warnings"].append("البيانات التاريخية لا تحتوي على تصنيفات عناقيد K-Means.")

        # --- 2. توصيات بناءً على موضوع BERTopic ---
        bertopic_id_val = problem_analysis_results.get("bertopic_topic")
        if bertopic_id_val is not None and pd.notna(bertopic_id_val) and int(
                bertopic_id_val) >= 0 and 'bertopic_topic' in self.historical_data.columns:
            bertopic_id = int(bertopic_id_val)  # تأكد أنه int
            print(f"\nالبحث عن توصيات بناءً على موضوع BERTopic رقم: {bertopic_id}")
            if current_problem_id is not None:
                similar_problems_b = self.historical_data[
                    (self.historical_data['bertopic_topic'] == bertopic_id) &
                    (self.historical_data['problem_id'] != current_problem_id)
                    ]
            else:
                similar_problems_b = self.historical_data[
                    (self.historical_data['bertopic_topic'] == bertopic_id)
                ]

            if not similar_problems_b.empty:
                print(f"تم العثور على {len(similar_problems_b)} مشكلة مشابهة في نفس موضوع BERTopic.")
                recommendations_output["based_on_bertopic_topic"] = self._extract_recommendations_from_df(
                    similar_problems_b, top_n)
            else:
                recommendations_output["general_warnings"].append(
                    f"لم يتم العثور على مشاكل أخرى في موضوع BERTopic رقم {bertopic_id} (باستثناء المشكلة الحالية إذا كان لها ID).")
        elif bertopic_id_val is not None and pd.notna(bertopic_id_val) and int(bertopic_id_val) < 0:  # مثل -1 أو -2
            recommendations_output["general_warnings"].append(
                f"المشكلة صُنفت كموضوع ضوضاء/غير محدد ({int(bertopic_id_val)}) بواسطة BERTopic، لا توجد توصيات موضوعية محددة.")
        elif 'bertopic_topic' not in self.historical_data.columns:
            recommendations_output["general_warnings"].append(
                "البيانات التاريخية لا تحتوي على تصنيفات موضوعات BERTopic.")

        if not recommendations_output["based_on_kmeans_cluster"] and not recommendations_output[
            "based_on_bertopic_topic"]:
            if not recommendations_output["general_warnings"]:
                recommendations_output["general_warnings"].append(
                    "لم يتم العثور على توصيات محددة بناءً على التصنيفات الحالية.")
        return recommendations_output


# --- مثال للاستخدام (للاختبار) ---
if __name__ == '__main__':
    print("===== بدء اختبار RecommendationEngine =====")
    sample_analysis_results_1 = {
        "input_problem_data": {'problem_id': 9901, 'title': 'الشبكة بطيئة جدا في قسم المحاسبة'},
        # ID مختلف للتأكد من عدم استبعاده
        "kmeans_cluster": 2,
        "bertopic_topic": 1,
    }
    sample_analysis_results_2 = {
        "input_problem_data": {'problem_id': 9902, 'title': 'طابعة لا تطبع'},
        "kmeans_cluster": 1,
        "bertopic_topic": -1,
    }

    if not os.path.exists(HISTORICAL_DATA_WITH_ALL_RESULTS_PATH):
        print(
            f"خطأ فادح: ملف البيانات التاريخية '{HISTORICAL_DATA_WITH_ALL_RESULTS_PATH}' غير موجود! لا يمكن اختبار RecommendationEngine.")
    else:
        recommender = RecommendationEngine()  # سيستخدم المسار الافتراضي
        if recommender.historical_data is not None:
            print("\n--- توليد توصيات للمشكلة 1 (شبكة بطيئة) ---")
            recs_1 = recommender.get_recommendations(sample_analysis_results_1)
            for category, rec_list in recs_1.items():
                if rec_list:
                    print(f"  {category}:")
                    for rec in rec_list:
                        print(f"    - {rec}")

            print("\n" + "=" * 30 + "\n")

            print("--- توليد توصيات للمشكلة 2 (طابعة لا تطبع) ---")
            recs_2 = recommender.get_recommendations(sample_analysis_results_2)
            for category, rec_list in recs_2.items():
                if rec_list:
                    print(f"  {category}:")
                    for rec in rec_list:
                        print(f"    - {rec}")
        else:
            print("فشل تحميل البيانات التاريخية. لا يمكن إجراء اختبار RecommendationEngine.")
    print("\n===== انتهاء اختبار RecommendationEngine =====")