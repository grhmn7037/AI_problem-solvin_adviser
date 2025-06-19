# src/data_processing/data_preprocessor.py
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import re  # استيراد re لاستخدامه في الدوال المساعدة

# تأكد من أن مسارات الاستيراد صحيحة
try:
    from src.data_processing.database_connector import DatabaseConnector
    from src.utils.text_processing import preprocess_text_pipeline
except ImportError:
    import sys

    current_dir_preprocessor = os.path.dirname(os.path.abspath(__file__))
    project_root_preprocessor = os.path.abspath(os.path.join(current_dir_preprocessor, '..', '..'))
    if project_root_preprocessor not in sys.path:
        sys.path.insert(0, project_root_preprocessor)
    from src.data_processing.database_connector import DatabaseConnector
    from src.utils.text_processing import preprocess_text_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- دوال مساعدة لتحويل القيم ---
def parse_cost_value(cost_str: str) -> float:
    """
    محاولة تحليل قيمة التكلفة من نص وتحويلها إلى رقم.
    """
    if pd.isna(cost_str) or cost_str == '':
        return np.nan

    cost_str = str(cost_str).lower()  # توحيد حالة الأحرف
    numeric_value = np.nan

    # 1. البحث عن الأرقام (بما في ذلك الكسور العشرية)
    #    مثال: "5 دولار", "20-50 ريال", "25000"
    numbers_found = re.findall(r'\d+\.?\d*', cost_str)

    if numbers_found:
        if len(numbers_found) == 1:
            numeric_value = float(numbers_found[0])
        elif len(numbers_found) > 1 and (
                '-' in cost_str or 'الى' in cost_str or 'إلى' in cost_str):  # التعامل مع النطاقات
            try:
                # أخذ متوسط النطاق
                numeric_value = (float(numbers_found[0]) + float(numbers_found[-1])) / 2
            except ValueError:
                numeric_value = float(numbers_found[0])  # إذا فشل، خذ الرقم الأول
        else:  # إذا كان هناك عدة أرقام غير مرتبطة بنطاق واضح، خذ الأول كافتراض
            numeric_value = float(numbers_found[0])

        # يمكن إضافة منطق تحويل عملات هنا إذا لزم الأمر لاحقًا
        # حاليًا، نحن فقط نستخلص القيمة الرقمية

    else:  # إذا لم يتم العثور على أرقام، تحقق من الكلمات الوصفية
        if "عالي" in cost_str or "مرتفع" in cost_str:
            numeric_value = 10000.0  # قيمة تقديرية لعالي (يمكن تعديلها)
        elif "متوسط" in cost_str:
            numeric_value = 5000.0  # قيمة تقديرية لمتوسط
        elif "منخفض" in cost_str:
            numeric_value = 1000.0  # قيمة تقديرية لمنخفض
        # يمكنك إضافة المزيد من الكلمات المفتاحية هنا

    return numeric_value


def parse_time_to_implement(time_str: str) -> float:
    """
    محاولة تحليل وقت التنفيذ من نص وتحويله إلى عدد الأيام.
    """
    if pd.isna(time_str) or time_str == '':
        return np.nan

    time_str = str(time_str).lower()
    days = np.nan

    # 1. التعامل مع "فوري"
    if "فوري" in time_str:
        return 0.0  # أو 1.0 إذا كنت تعتبر "فوري" يستغرق يومًا واحدًا كحد أدنى

    # 2. البحث عن الأرقام والوحدات
    numbers_found = re.findall(r'\d+\.?\d*', time_str)

    value1 = None
    value2 = None

    if numbers_found:
        value1 = float(numbers_found[0])
        if len(numbers_found) > 1 and ('-' in time_str or 'الى' in time_str or 'إلى' in time_str):
            value2 = float(numbers_found[-1])
            # أخذ متوسط النطاق
            avg_value = (value1 + value2) / 2
        else:
            avg_value = value1
    else:  # لا توجد أرقام، قد يكون نصًا وصفيًا آخر غير "فوري"
        return np.nan

    # تحديد الوحدة وتحويلها إلى أيام
    if "شهر" in time_str or "اشهر" in time_str or "أشهر" in time_str:
        days = avg_value * 30  # افتراض الشهر 30 يومًا
    elif "اسبوع" in time_str or "أسبوع" in time_str or "اسابيع" in time_str or "أسابيع" in time_str:
        days = avg_value * 7
    elif "يوم" in time_str or "ايام" in time_str or "أيام" in time_str:
        days = avg_value
    elif "ساعه" in time_str or "ساعة" in time_str or "ساعات" in time_str:
        days = avg_value / 24
    elif "دقيقه" in time_str or "دقيقة" in time_str or "دقائق" in time_str:
        days = avg_value / (24 * 60)
    else:  # إذا لم يتم تحديد وحدة واضحة وكان هناك رقم، نفترض أنها أيام (أو نتركها NaN)
        # هذا يعتمد على السياق، قد يكون من الأفضل تركها NaN إذا لم تكن الوحدة واضحة
        # days = avg_value # افتراض أيام إذا لم تذكر وحدة
        return np.nan  # أكثر أمانًا إذا لم تكن الوحدة واضحة

    return days


class DataPreprocessor:
    def __init__(self, db_connector: DatabaseConnector):
        self.db_connector = db_connector
        self.raw_data = None
        self.processed_data = None

    def load_data(self, limit: int = None) -> pd.DataFrame:
        logging.info("بدء تحميل البيانات الخام...")
        try:
            self.raw_data = self.db_connector.extract_problems_data(limit=limit)
            logging.info(f"تم تحميل {len(self.raw_data)} سجل خام بنجاح.")
            logging.info(f"أبعاد البيانات الخام: {self.raw_data.shape}")
            # logging.info(f"أول 3 صفوف من البيانات الخام:\n{self.raw_data.head(3)}")
            # logging.info(f"معلومات الأعمدة وأنواع البيانات الأولية:\n{self.raw_data.info()}")
            return self.raw_data
        except Exception as e:
            logging.error(f"خطأ أثناء تحميل البيانات الخام: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("بدء معالجة القيم المفقودة...")
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna('')
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

        # معالجة خاصة لـ sentiment_score إذا كان مفقودًا بعد التحويل الرقمي
        if 'sentiment_score' in df.columns and pd.api.types.is_numeric_dtype(df['sentiment_score']):
            df['sentiment_score'] = df['sentiment_score'].fillna(0.0)  # ملء NaN بالصفر للمشاعر المحايدة

        logging.info("اكتملت معالجة القيم المفقودة.")
        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("بدء تحويل أنواع البيانات...")

        date_columns = ['date_identified', 'date_closed', 'date_chosen',
                        'start_date_planned', 'end_date_planned',
                        'start_date_actual', 'end_date_actual']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logging.info(f"تم تحويل العمود '{col}' إلى datetime.")

        if 'sentiment_score' in df.columns and not pd.api.types.is_numeric_dtype(df['sentiment_score']):
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            logging.info("تم تحويل العمود 'sentiment_score' إلى رقمي.")

        # --- تحويل الأعمدة المالية والزمنية ---
        if 'estimated_cost' in df.columns:
            df['estimated_cost_numeric'] = df['estimated_cost'].apply(parse_cost_value)
            logging.info("تم إنشاء العمود 'estimated_cost_numeric'.")

        if 'overall_budget' in df.columns:
            df['overall_budget_numeric'] = df['overall_budget'].apply(parse_cost_value)
            logging.info("تم إنشاء العمود 'overall_budget_numeric'.")

        if 'estimated_time_to_implement' in df.columns:
            df['estimated_time_days'] = df['estimated_time_to_implement'].apply(parse_time_to_implement)
            logging.info("تم إنشاء العمود 'estimated_time_days'.")

        logging.info("اكتمل تحويل أنواع البيانات.")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("بدء هندسة الميزات...")

        if 'date_identified' in df.columns and 'date_closed' in df.columns:
            df['resolution_time_days_calc'] = (df['date_closed'] - df['date_identified']).dt.days
            df['resolution_time_days_calc'] = df['resolution_time_days_calc'].apply(
                lambda x: x if pd.notna(x) and x >= 0 else np.nan)
            # تم تغيير اسم العمود لتجنب التعارض مع عمود resolution_time_days الأصلي إذا كان موجودًا بمعنى مختلف
            logging.info("تم إنشاء الميزة 'resolution_time_days_calc'.")

        text_fields_to_combine = [
            'title', 'description_initial', 'refined_problem_statement_final',
            'stakeholders_involved', 'initial_impact_assessment', 'problem_source',
            'active_listening_notes', 'key_questions_asked', 'initial_hypotheses',
            'key_findings_from_analysis', 'potential_root_causes_list',
            'solution_description', 'justification_for_choice',
            'what_went_well', 'what_could_be_improved', 'recommendations_for_future', 'key_takeaways'
        ]
        existing_text_fields = [col for col in text_fields_to_combine if col in df.columns]
        logging.info(f"الأعمدة النصية التي سيتم دمجها: {existing_text_fields}")
        df['combined_text_for_nlp'] = df[existing_text_fields].astype(str).agg(' '.join, axis=1)
        logging.info("تم إنشاء الميزة 'combined_text_for_nlp'.")

        logging.info("بدء تطبيق تنظيف النصوص على 'combined_text_for_nlp'...")
        df['processed_text'] = df['combined_text_for_nlp'].apply(
            lambda x: preprocess_text_pipeline(x, use_stemming=False))
        logging.info("اكتمل تنظيف النصوص لـ 'processed_text'.")

        logging.info("اكتملت هندسة الميزات.")
        return df

    def preprocess(self, limit: int = None, save_processed_data: bool = True,
                   processed_data_path: str = "data/processed/processed_problems_data.csv") -> pd.DataFrame:
        self.load_data(limit=limit)
        if self.raw_data is None or self.raw_data.empty:
            logging.error("لا توجد بيانات خام للمعالجة.")
            return pd.DataFrame()

        df = self.raw_data.copy()
        # 1. تحويل أنواع البيانات (بما في ذلك الأعمدة المالية والزمنية) - يُفضل قبل ملء القيم المفقودة للأعمدة الجديدة
        df = self._convert_data_types(df)
        # 2. معالجة القيم المفقودة (للأعمدة الأصلية والجديدة التي قد تحتوي على NaN بعد التحويل)
        df = self._handle_missing_values(df)
        # 3. هندسة الميزات (بما في ذلك معالجة النصوص)
        df = self._engineer_features(df)

        self.processed_data = df
        logging.info("اكتملت جميع خطوات المعالجة المسبقة.")
        logging.info(f"أبعاد البيانات المعالجة: {self.processed_data.shape}")

        cols_to_show = ['problem_id', 'title', 'processed_text',
                        'estimated_cost_numeric', 'overall_budget_numeric', 'estimated_time_days',
                        'resolution_time_days_calc']
        # تأكد أن الأعمدة موجودة قبل محاولة عرضها
        existing_cols_to_show = [col for col in cols_to_show if col in self.processed_data.columns]
        logging.info(
            f"أول 3 صفوف من البيانات المعالجة (أعمدة مختارة):\n{self.processed_data[existing_cols_to_show].head(3)}")

        # logging.info(f"معلومات الأعمدة وأنواع البيانات النهائية:\n{self.processed_data.info()}")

        if save_processed_data:
            try:
                os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                self.processed_data.to_csv(processed_data_path, index=False, encoding='utf-8-sig')
                logging.info(f"تم حفظ البيانات المعالجة في: {processed_data_path}")
            except Exception as e:
                logging.error(f"خطأ أثناء حفظ البيانات المعالجة: {e}")
        return self.processed_data


# مثال للاختبار
if __name__ == '__main__':
    try:
        db_connector_instance = DatabaseConnector()
        preprocessor = DataPreprocessor(db_connector=db_connector_instance)
        processed_df = preprocessor.preprocess(limit=None)  # معالجة جميع البيانات هذه المرة

        if not processed_df.empty:
            print("\n--- عينة من البيانات المعالجة (الأعمدة الجديدة) ---")
            cols_to_print = ['problem_id', 'estimated_cost', 'estimated_cost_numeric',
                             'overall_budget', 'overall_budget_numeric',
                             'estimated_time_to_implement', 'estimated_time_days',
                             'resolution_time_days_calc', 'processed_text']
            existing_cols_to_print = [col for col in cols_to_print if col in processed_df.columns]
            print(processed_df[existing_cols_to_print].to_string())  # to_string لعرض كل الصفوف والأعمدة

            # print("\n--- معلومات البيانات المعالجة ---")
            # processed_df.info()
        else:
            print("لم يتم إنتاج أي بيانات معالجة.")

    except FileNotFoundError as e:
        logging.error(f"خطأ في مسار ملف قاعدة البيانات، تأكد من إعدادات config: {e}")
    except ConnectionError as e:
        logging.error(f"خطأ في الاتصال بقاعدة البيانات: {e}")
    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع أثناء اختبار DataPreprocessor: {e}", exc_info=True)
    finally:
        if 'db_connector_instance' in locals() and hasattr(db_connector_instance,
                                                           'engine') and db_connector_instance.engine:
            db_connector_instance.close_connection()