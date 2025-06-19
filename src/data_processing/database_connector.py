# src/data_processing/database_connector.py
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Optional
import logging
import os

# تعديل لاستيراد المسار من ملف الإعدادات الجديد
# تأكد أن config موجود في PYTHONPATH أو استخدم مسار نسبي صحيح
try:
    from config.database_config import SQLITE_DB_PATH, TABLES
except ImportError:
    # هذا المسار البديل قد يعمل إذا كنت تشغل الملف مباشرة من مجلده
    # ويتطلب أن يكون مجلد config في نفس مستوى مجلد src أو أن يكون PYTHONPATH معد بشكل صحيح
    # من الأفضل دائمًا تشغيل الأكواد من جذر المشروع.
    import sys
    # إضافة المسار الجذر للمشروع إلى sys.path
    # يفترض أن هذا الملف موجود في problem_ai_advisor/src/data_processing/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.database_config import SQLITE_DB_PATH, TABLES


class DatabaseConnector:
    """
    كلاس للاتصال بقاعدة بيانات SQLite واستخراج البيانات باستخدام SQLAlchemy.
    """

    def __init__(self, db_path: str = None):
        """
        تهيئة الاتصال بقاعدة البيانات.
        Args:
            db_path (str, optional): المسار إلى ملف قاعدة بيانات SQLite.
                                     إذا لم يتم توفيره، سيتم استخدام المسار من ملف الإعدادات.
        """
        self.db_path = db_path or SQLITE_DB_PATH
        self.engine = None
        self._connect() # الاتصال عند الإنشاء

    def _connect(self):
        """
        إنشاء اتصال بقاعدة البيانات.
        """
        try:
            if not os.path.exists(self.db_path):
                logging.error(f"ملف قاعدة البيانات غير موجود: {self.db_path}")
                # يمكنك إما إثارة استثناء هنا أو ترك self.engine = None ليتم التعامل معه لاحقًا
                raise FileNotFoundError(f"ملف قاعدة البيانات غير موجود: {self.db_path}")

            # SQLAlchemy يتطلب بادئة 'sqlite:///' لملفات SQLite
            conn_string = f"sqlite:///{self.db_path}"
            self.engine = create_engine(conn_string)
            # اختبار الاتصال
            with self.engine.connect() as connection:
                logging.info(f"تم الاتصال بقاعدة بيانات SQLite بنجاح: {self.db_path}")

        except SQLAlchemyError as e:
            logging.error(f"خطأ SQLAlchemy في الاتصال بقاعدة البيانات ({self.db_path}): {e}")
            self.engine = None # تأكد من أن المحرك لا يزال None في حالة الفشل
            raise # إعادة إثارة الاستثناء للسماح للمستدعي بالتعامل معه
        except FileNotFoundError as e:
            logging.error(f"خطأ FileNotFoundError: {e}")
            self.engine = None
            raise
        except Exception as e:
            logging.error(f"خطأ عام في الاتصال بقاعدة البيانات ({self.db_path}): {e}")
            self.engine = None
            raise

    def _ensure_connected(self):
        """يتأكد من وجود اتصال صالح، ويحاول إعادة الاتصال إذا لزم الأمر."""
        if self.engine is None:
            logging.warning("لا يوجد اتصال بالمحرك. محاولة إعادة الاتصال...")
            self._connect()
        if self.engine is None: # إذا فشلت إعادة الاتصال
             raise ConnectionError("فشل الاتصال بقاعدة البيانات. المحرك غير متاح.")


    def extract_data(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        تنفيذ استعلام SQL واستخراج البيانات كـ DataFrame.
        Args:
            query (str): استعلام SQL.
            params (Optional[Dict]): معاملات للاستعلام (للأمان ضد SQL Injection).
        Returns:
            pd.DataFrame: البيانات المستخرجة.
        """
        self._ensure_connected()
        try:
            # استخدام with self.engine.connect() يضمن إغلاق الاتصال تلقائيًا
            with self.engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection, params=params)
            logging.info(f"تم استخراج {len(df)} سجل بنجاح من الاستعلام.")
            return df
        except SQLAlchemyError as e:
            logging.error(f"خطأ SQLAlchemy أثناء استخراج البيانات: {e}")
            raise
        except Exception as e:
            logging.error(f"خطأ عام أثناء استخراج البيانات: {e}")
            raise # أو إرجاع DataFrame فارغ: return pd.DataFrame()

    def extract_problems_data(self, limit: int = None) -> pd.DataFrame:
        """
        استخراج بيانات المشاكل مع المعلومات المرتبطة بها كما في الكود الأصلي.
        """
        # الاستعلام الذي قدمته يبدو جيدًا وشاملاً.
        # تأكد من أن جميع أسماء الجداول والأعمدة تتطابق تمامًا مع مخطط قاعدة بيانات SQLite.
        # SQLite قد يكون حساسًا لحالة الأحرف بشكل مختلف عن PostgreSQL في بعض الإعدادات.
        # GROUP_CONCAT متاح في SQLite، وهو جيد.
        query = """
        SELECT
            p.id AS problem_id,
            p.title,
            p.description_initial,
            p.domain,
            p.complexity_level,
            p.date_identified,
            p.date_closed,
            p.status,
            p.stakeholders_involved,
            p.initial_impact_assessment,
            p.problem_source,
            p.refined_problem_statement_final,
            p.sentiment_score,
            p.sentiment_label,
            p.problem_tags,
            p.ai_generated_summary,
            pu.active_listening_notes,
            pu.key_questions_asked,
            pu.initial_data_sources,
            pu.initial_hypotheses,
            pu.stakeholder_feedback_initial,
            ca.data_collection_methods_deep,
            ca.data_analysis_techniques_used,
            ca.key_findings_from_analysis,
            cs.justification_for_choice,
            cs.approval_status,
            cs.date_chosen,
            ps.solution_description,
            ps.generation_method,
            ps.estimated_cost,
            ps.estimated_time_to_implement,
            ps.potential_benefits,
            ps.potential_risks,
            ip.plan_description,
            ip.overall_status as implementation_status,
            ip.start_date_planned,
            ip.end_date_planned,
            ip.start_date_actual,
            ip.end_date_actual,
            ip.overall_budget,
            ip.key_personnel,
            ll.what_went_well,
            ll.what_could_be_improved,
            ll.recommendations_for_future,
            ll.key_takeaways,
            (SELECT GROUP_CONCAT(prc.cause_description, '; ')
             FROM potential_root_cause prc
             WHERE prc.analysis_id = ca.id
            ) AS potential_root_causes_list -- SQLite قد يحتاج هذا في استعلام فرعي إذا كان الربط معقدًا
            -- ملاحظة: في الكود الأصلي، ps.id مرتبط بـ csol.proposed_solution_id
            -- وهو ما تم استخدامه هنا.
        FROM problem p
        LEFT JOIN problem_understanding pu ON p.id = pu.problem_id
        LEFT JOIN cause_analysis ca ON p.id = ca.problem_id
        LEFT JOIN chosen_solution cs ON p.id = cs.problem_id
        LEFT JOIN proposed_solution ps ON cs.proposed_solution_id = ps.id -- ربط الحل المقترح من خلال الحل المختار
        LEFT JOIN implementation_plan ip ON cs.id = ip.chosen_solution_id
        LEFT JOIN lesson_learned ll ON p.id = ll.problem_id
        -- لا نحتاج GROUP BY p.id إذا كان كل مشكلة لها بالكثير صف واحد من كل جدول مرتبط (علاقة واحد لواحد أو واحد لكثير مع اختيار واحد)
        -- إذا كانت هناك علاقات كثير لكثير قد تؤدي لتكرار، ستحتاج GROUP BY p.id وربما GROUP_CONCAT لباقي الحقول النصية المجمعة
        ORDER BY p.id -- جيد للاتساق
        """
        # تعديل محتمل: إذا كان الربط بـ potential_root_cause من خلال cause_analysis يؤدي لصفوف متعددة للمشكلة الواحدة،
        # ستحتاج إلى GROUP BY p.id واستخدام GROUP_CONCAT للحقول النصية من الجداول المربوطة (مثل ps.solution_description إذا كان يمكن أن يكون هناك أكثر من حل مقترح مرتبط بطريقة ما قبل الاختيار).
        # ومع ذلك، بناءً على `cs.proposed_solution_id = ps.id`، يبدو أنك تربط حلاً مقترحًا *واحدًا* محددًا تم اختياره.

        if limit:
            query += f" LIMIT {limit}"

        return self.extract_data(query)

    def extract_kpi_data(self) -> pd.DataFrame:
        """
        استخراج بيانات مؤشرات الأداء.
        """
        query = """
        SELECT
            sk.chosen_solution_id,
            sk.kpi_name,
            sk.kpi_description,
            sk.target_value,
            sk.current_value_baseline,
            sk.measurement_unit,
            sk.measurement_frequency,
            km.measurement_date,
            km.actual_value,
            km.notes
        FROM solution_kpi sk
        LEFT JOIN kpi_measurement km ON sk.id = km.kpi_id
        ORDER BY sk.chosen_solution_id, km.measurement_date
        """
        return self.extract_data(query)

    def extract_root_causes(self) -> pd.DataFrame:
        """
        استخراج بيانات الأسباب الجذرية.
        """
        query = """
        SELECT
            prc.analysis_id,
            ca.problem_id,
            prc.cause_description,
            prc.evidence_supporting_cause,
            prc.validation_status,
            prc.impact_of_cause
        FROM potential_root_cause prc
        JOIN cause_analysis ca ON prc.analysis_id = ca.id
        """
        return self.extract_data(query)

    def get_database_stats(self) -> Dict:
        """
        الحصول على إحصائيات قاعدة البيانات.
        """
        self._ensure_connected()
        stats = {}
        queries = {
            'total_problems': "SELECT COUNT(*) as count FROM problem",
            'total_solutions_proposed': "SELECT COUNT(*) as count FROM proposed_solution", # تم تغيير الاسم ليعكس الجدول
            'total_solutions_chosen': "SELECT COUNT(*) as count FROM chosen_solution",
            'solved_problems': "SELECT COUNT(*) as count FROM problem WHERE status = 'Closed'", # تأكد من القيمة الدقيقة لـ 'closed'
            'unique_domains': "SELECT COUNT(DISTINCT domain) as count FROM problem WHERE domain IS NOT NULL AND domain != ''"
        }
        try:
            with self.engine.connect() as connection:
                for key, query_str in queries.items():
                    result = connection.execute(text(query_str)).fetchone()
                    stats[key] = result[0] if result else 0

            logging.info(f"إحصائيات قاعدة البيانات: {stats}")
            return stats

        except SQLAlchemyError as e:
            logging.error(f"خطأ SQLAlchemy في الحصول على الإحصائيات: {e}")
            raise
        except Exception as e:
            logging.error(f"خطأ عام في الحصول على الإحصائيات: {e}")
            raise

    def close_connection(self):
        """
        إغلاق الاتصال بقاعدة البيانات (التخلص من المحرك).
        """
        if self.engine:
            self.engine.dispose()
            logging.info("تم التخلص من محرك قاعدة البيانات.")
            self.engine = None

# مثال على الاستخدام (للاختبار السريع)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # إنشاء اتصال بقاعدة البيانات (سيستخدم المسار من config)
        db_connector = DatabaseConnector()

        # الحصول على إحصائيات
        stats = db_connector.get_database_stats()
        print("\nإحصائيات قاعدة البيانات:", stats)

        # استخراج بيانات المشاكل
        print("\nاستخراج بيانات المشاكل (بحد أقصى 5 مشاكل للاختبار):")
        problems_df = db_connector.extract_problems_data(limit=5)
        if not problems_df.empty:
            print(f"تم استخراج {len(problems_df)} مشكلة.")
            print("أعمدة البيانات:", problems_df.columns.tolist())
            print("أول بضعة صفوف:")
            print(problems_df.head())
            # التحقق من عمود الأسباب الجذرية المجمعة
            if 'potential_root_causes_list' in problems_df.columns:
                print("\nعينة من الأسباب الجذرية المجمعة:")
                print(problems_df[['problem_id', 'potential_root_causes_list']].head())
        else:
            print("لم يتم استخراج أي بيانات للمشاكل.")

        # استخراج بيانات مؤشرات الأداء
        print("\nاستخراج بيانات مؤشرات الأداء (بحد أقصى 5 للاختبار):")
        # لتجنب استخراج كل شيء، سنقوم بتعديل الاستعلام هنا مؤقتًا للاختبار إذا لم يكن لديك بيانات كثيرة
        # أو قم بإنشاء دالة مشابهة لـ extract_problems_data مع حد
        # kpi_df = db_connector.extract_data("SELECT * FROM solution_kpi sk LEFT JOIN kpi_measurement km ON sk.id = km.kpi_id LIMIT 5")
        kpi_df = db_connector.extract_kpi_data() # إذا كانت الدالة موجودة وتدعم LIMIT أو إذا كانت البيانات قليلة
        if not kpi_df.empty:
             print(f"تم استخراج {len(kpi_df)} سجل KPI.")
             print(kpi_df.head())
        else:
            print("لم يتم استخراج أي بيانات لمؤشرات الأداء.")


    except FileNotFoundError as e:
        logging.error(f"خطأ في مسار الملف: {e}")
    except ConnectionError as e:
        logging.error(f"خطأ في الاتصال: {e}")
    except SQLAlchemyError as e:
        logging.error(f"خطأ SQLAlchemy: {e}")
    except Exception as e:
        logging.error(f"حدث خطأ غير متوقع: {e}", exc_info=True) # طباعة تتبع الخطأ الكامل
    finally:
        # إغلاق الاتصال (إذا تم فتحه بنجاح)
        if 'db_connector' in locals() and db_connector.engine:
            db_connector.close_connection()