# src/models/clustering_model.py
import pandas as pd
import numpy as np
import os
import joblib
from scipy.sparse import hstack, csr_matrix  # لا يزال مفيدًا إذا كان CT ينتج متفرقًا

# *** استيراد جديد ***
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("تحذير: مكتبة SentenceTransformer غير مثبتة. pip install sentence-transformers")
    SentenceTransformer = None

# المسارات الافتراضية للمكونات الجديدة
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_MODELS_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'data', 'models')
DEFAULT_KMEANS_PATH = os.path.join(DEFAULT_MODELS_DIR, 'kmeans_model.pkl')  # نفس اسم ملف النموذج
# *** اسم ملف ColumnTransformer الجديد الذي يعالج الميزات الرقمية والفئوية فقط ***
DEFAULT_CT_PATH = os.path.join(DEFAULT_MODELS_DIR, 'ct_num_cat_embeddings_preprocessor.pkl')


# لم نعد نحتاج إلى TFIDF_VECTORIZER_PATH هنا

class ProblemClusteringModel:
    def __init__(self,
                 kmeans_model_path: str = DEFAULT_KMEANS_PATH,
                 ct_preprocessor_path: str = DEFAULT_CT_PATH,
                 embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        تهيئة نموذج التجميع.
        Args:
            kmeans_model_path (str): مسار ملف نموذج K-Means المحفوظ.
            ct_preprocessor_path (str): مسار ملف ColumnTransformer (للميزات الرقمية/الفئوية) المحفوظ.
            embedding_model_name (str): اسم أو مسار نموذج تضمين الجمل من SentenceTransformer.
        """
        self.kmeans_model = None
        self.column_transformer = None
        self.sentence_model = None  # *** كائن لنموذج التضمين ***
        self.embedding_model_name = embedding_model_name

        self.numerical_features = []
        self.categorical_features = []
        self.text_feature_col = 'processed_text'  # العمود الذي يحتوي على النصوص النظيفة


        try:
            print(f"محاولة تحميل نموذج K-Means من: {kmeans_model_path}")
            self.kmeans_model = joblib.load(kmeans_model_path)
            print("تم تحميل نموذج K-Means بنجاح.")

            print(f"محاولة تحميل ColumnTransformer (num/cat) من: {ct_preprocessor_path}")
            self.column_transformer = joblib.load(ct_preprocessor_path)
            print("تم تحميل ColumnTransformer (num/cat) بنجاح.")
            self._extract_feature_names_from_ct()

            if SentenceTransformer:
                print(f"محاولة تحميل نموذج تضمين الجمل: {self.embedding_model_name}")
                self.sentence_model = SentenceTransformer(self.embedding_model_name)
                print("تم تحميل نموذج تضمين الجمل بنجاح.")
            else:
                print("خطأ: مكتبة SentenceTransformer غير متاحة. لا يمكن تحميل نموذج التضمين.")
                # قد ترغب في إثارة استثناء هنا إذا كان هذا حرجًا

            print("اكتمل تحميل جميع مكونات نموذج التجميع (بما في ذلك نموذج التضمين).")

        except FileNotFoundError as e:
            print(f"خطأ في تحميل المكونات: ملف غير موجود - {e}")
        except Exception as e:
            print(f"خطأ عام أثناء تحميل مكونات النموذج: {e}")

    def _extract_feature_names_from_ct(self):
        if self.column_transformer:
            try:
                num_transformer_tuple = next(t for t in self.column_transformer.transformers_ if t[0] == 'num')
                self.numerical_features = num_transformer_tuple[2]
                print(f"الميزات الرقمية المستخلصة من CT: {self.numerical_features}")

                cat_transformer_tuple = next(t for t in self.column_transformer.transformers_ if t[0] == 'cat')
                self.categorical_features = cat_transformer_tuple[2]
                print(f"الميزات الفئوية المستخلصة من CT: {self.categorical_features}")
            except StopIteration:
                print("تحذير: لم يتم العثور على محولات 'num' أو 'cat' بالأسماء المتوقعة في ColumnTransformer.")
            except Exception as e:
                print(f"خطأ أثناء استخلاص أسماء الميزات من ColumnTransformer: {e}")

    def _preprocess_single_problem_data(self, problem_data_df: pd.DataFrame) -> pd.DataFrame:
        df = problem_data_df.copy()
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(
                    0)  # ملء بسيط بـ 0 (يجب أن يكون StandardScaler قد تم تدريبه على بيانات لا تحتوي NaN)
                print(f"تم ملء NaN في '{col}' (رقمي) بـ 0 للتنبؤ.")
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                if not pd.api.types.is_string_dtype(df[col]) and pd.api.types.is_object_dtype(df[col]):
                    df[col] = df[col].astype(str)
                df[col] = df[col].fillna('Unknown')
                print(f"تم ملء NaN في '{col}' (فئوي) بـ 'Unknown' للتنبؤ.")
        # لا نحتاج لملء text_feature_col هنا لأن SentenceTransformer سيتعامل مع النصوص الفارغة
        return df

    def predict(self, new_problems_df: pd.DataFrame) -> np.ndarray:
        if not all([self.kmeans_model, self.column_transformer, self.sentence_model]):
            print("خطأ: النموذج أو أحد مكونات المعالجة/التضمين غير محمل. لا يمكن التنبؤ.")
            return np.array([])
        if not isinstance(new_problems_df, pd.DataFrame) or new_problems_df.empty:
            print("خطأ: البيانات المدخلة يجب أن تكون DataFrame صالح وغير فارغ.")
            return np.array([])

        print(f"\nبدء معالجة {len(new_problems_df)} مشكلة جديدة للتنبؤ (باستخدام تضمينات الجمل)...")

        required_cols_for_ct = self.numerical_features + self.categorical_features
        missing_ct_cols = [col for col in required_cols_for_ct if col not in new_problems_df.columns]
        if missing_ct_cols:
            print(f"خطأ: أعمدة مفقودة لـ ColumnTransformer: {missing_ct_cols}")
            return np.array([])
        if self.text_feature_col not in new_problems_df.columns:
            print(f"خطأ: العمود النصي '{self.text_feature_col}' مفقود.")
            return np.array([])

        df_preprocessed_light = self._preprocess_single_problem_data(new_problems_df)

        try:
            num_cat_features_transformed = self.column_transformer.transform(df_preprocessed_light)
            print(f"تم تطبيق ColumnTransformer (num/cat). أبعاد الميزات: {num_cat_features_transformed.shape}")
        except Exception as e:
            print(f"خطأ أثناء تطبيق ColumnTransformer: {e}")
            return np.array([])

        print(f"إنشاء تضمينات للنصوص الجديدة باستخدام: {self.embedding_model_name}...")
        texts_to_embed_new = df_preprocessed_light[self.text_feature_col].astype(str).tolist()
        text_embeddings_new = self.sentence_model.encode(texts_to_embed_new,
                                                         show_progress_bar=False)  # لا حاجة لشريط تقدم لمشكلة واحدة عادة
        print(f"تم إنشاء تضمينات النصوص. أبعاد مصفوفة التضمينات: {text_embeddings_new.shape}")

        # دمج الميزات: نفترض أن num_cat_features_transformed مصفوفة numpy كثيفة
        # وأن text_embeddings_new مصفوفة numpy كثيفة
        try:
            # تأكد أن كلاهما 2D arrays
            if num_cat_features_transformed.ndim == 1: num_cat_features_transformed = num_cat_features_transformed.reshape(
                1, -1)
            if text_embeddings_new.ndim == 1: text_embeddings_new = text_embeddings_new.reshape(1, -1)

            final_features_for_prediction = np.concatenate([num_cat_features_transformed, text_embeddings_new], axis=1)
            print(f"تم دمج الميزات (كثيفة). الأبعاد النهائية: {final_features_for_prediction.shape}")
        except ValueError as ve:
            print(f"خطأ في أبعاد المصفوفات أثناء الدمج: {ve}")
            print(f"  أبعاد num_cat: {num_cat_features_transformed.shape}")
            print(f"  أبعاد text_embed: {text_embeddings_new.shape}")
            return np.array([])

        if hasattr(self.kmeans_model, 'n_features_in_') and \
                final_features_for_prediction.shape[1] != self.kmeans_model.n_features_in_:
            print(f"خطأ: عدد الميزات في البيانات الجديدة ({final_features_for_prediction.shape[1]}) "
                  f"لا يتطابق مع عدد الميزات التي تم تدريب النموذج عليها ({self.kmeans_model.n_features_in_}).")
            return np.array([])

        print("التنبؤ بتسميات العناقيد...")
        cluster_predictions = self.kmeans_model.predict(final_features_for_prediction)
        print(f"تم التنبؤ بـ {len(cluster_predictions)} عنقود(عناقيد).")
        return cluster_predictions


# --- مثال للاستخدام (للاختبار) ---
if __name__ == '__main__':
    print("--- بدء اختبار ProblemClusteringModel (مع تضمينات الجمل) ---")

    clustering_model_embed = ProblemClusteringModel()

    if clustering_model_embed.kmeans_model and clustering_model_embed.column_transformer and clustering_model_embed.sentence_model:
        print("\n--- إعداد بيانات اختبار مشابهة لما لدينا ---")

        if clustering_model_embed.numerical_features and clustering_model_embed.categorical_features:
            sample_problem_data = {
                # عمود النص النظيف مطلوب
                'processed_text': ["مرحبا العالم هذا نص اختباري لمشكله جديده"],
                # قيم للميزات الرقمية (قبل التحجيم)
                'estimated_cost_numeric': [100.0],
                'overall_budget_numeric': [150.0],
                'estimated_time_days': [10.0],
                'processed_text_length': [7],  # إذا كانت هذه الميزة لا تزال مستخدمة
                # قيم للميزات الفئوية (القيم الأصلية)
                'domain': ['تقني'],
                'complexity_level': ['بسيط'],
                'status': ['مفتوحة'],
                'problem_source': ['ملاحظة'],
                'sentiment_label': ['محايد']
            }

            # التأكد من أن جميع الميزات المطلوبة بواسطة CT موجودة
            all_ct_input_cols = clustering_model_embed.numerical_features + clustering_model_embed.categorical_features
            for col in all_ct_input_cols:
                if col not in sample_problem_data:
                    print(
                        f"تحذير اختبار: العمود '{col}' متوقع بواسطة CT ولكنه غير موجود في بيانات الاختبار. سيتم استخدام NaN/None.")
                    sample_problem_data[col] = [np.nan if col in clustering_model_embed.numerical_features else None]

            df_new_sample = pd.DataFrame(sample_problem_data)

            print("\nبيانات المشكلة الجديدة (عينة):")
            # طباعة الأعمدة ذات الصلة فقط
            cols_to_show_test = [clustering_model_embed.text_feature_col] + \
                                [col for col in clustering_model_embed.numerical_features if
                                 col in df_new_sample.columns] + \
                                [col for col in clustering_model_embed.categorical_features if
                                 col in df_new_sample.columns]
            print(df_new_sample[cols_to_show_test].to_string())

            predictions = clustering_model_embed.predict(df_new_sample)

            if predictions.size > 0:
                print(f"\nالعنقود المتوقع للمشكلة الجديدة: {predictions[0]}")
        else:
            print("لم يتم استخلاص أسماء الميزات الرقمية/الفئوية من ColumnTransformer. لا يمكن إنشاء بيانات اختبار.")
    else:
        print("\nلم يتم تحميل جميع مكونات النموذج أو نموذج التضمين. لا يمكن إجراء اختبار التنبؤ.")

    print("\n--- انتهاء اختبار ProblemClusteringModel (مع تضمينات الجمل) ---")