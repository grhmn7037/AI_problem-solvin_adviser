# config/database_config.py
import os

# لتحديد مسار قاعدة البيانات بشكل أكثر ديناميكية، نفترض أن:
# - مشروع 'problem_ai_advisor'
# - ومشروع 'problem_management_ststem'
# موجودان داخل مجلد رئيسي مشترك (مثلاً 'pythonProject').

# المسار الحالي لملف config.py
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__)) # problem_ai_advisor/config/

# المسار الجذر لمشروع problem_ai_advisor
PROJECT_AI_ADVISOR_ROOT = os.path.dirname(CONFIG_DIR) # problem_ai_advisor/

# المسار إلى المجلد الذي يحتوي على كلا المشروعين
# نفترض أنه المجلد 'pythonProject' الذي ذكرته في مسار قاعدة البيانات
# إذا كان الهيكل مختلفًا، ستحتاج لتعديل هذا الجزء
PROJECTS_PARENT_DIR = os.path.dirname(os.path.dirname(PROJECT_AI_ADVISOR_ROOT))

# المسار الكامل لملف قاعدة البيانات
# تم تعديل هذا السطر ليعكس المسار الصحيح بناءً على ما قدمته
# C:\Users\pc\PycharmProjects\pythonProject\problem_management_ststem\instance\problem_management.db
# نفترض أن 'pythonProject' هو PROJECTS_PARENT_DIR
# وأن اسم المستخدم 'pc' صحيح. إذا كان يتغير، قد تحتاج إلى جعله أكثر مرونة أو استخدام مسار مطلق ثابت.

# الخيار الأول: استخدام مسار مطلق مباشرة (أسهل إذا كان المسار ثابتًا)
# SQLITE_DB_PATH = r"C:\Users\pc\PycharmProjects\pythonProject\problem_management_ststem\instance\problem_management.db"

# الخيار الثاني: محاولة بناء المسار بشكل نسبي (أكثر تعقيدًا إذا كانت المشاريع في أماكن مختلفة جدًا)
# هذا يفترض أن هيكل المجلدات كما هو موضح أعلاه
# إذا كان 'problem_management_ststem' داخل 'pythonProject' مباشرة:
DB_PROJECT_NAME = 'problem_management_ststem'
INSTANCE_FOLDER = 'instance'
DB_FILENAME = 'problem_management.db'

# افترض أن 'pythonProject' هو المجلد الأب لـ 'problem_ai_advisor'
# لذا، سنصعد مستوى واحد من PROJECT_AI_ADVISOR_ROOT للوصول إلى 'pythonProject'
PYTHON_PROJECT_DIR = os.path.dirname(PROJECT_AI_ADVISOR_ROOT)

SQLITE_DB_PATH = os.path.join(PYTHON_PROJECT_DIR, DB_PROJECT_NAME, INSTANCE_FOLDER, DB_FILENAME)


# للتأكد من المسار (يمكنك طباعته عند الاختبار)
# print(f"مسار قاعدة البيانات المحسوب: {SQLITE_DB_PATH}")


# لا نحتاج DATABASE_CONFIG لـ PostgreSQL حاليًا
# DATABASE_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'database': 'problem_management',
#     'username': 'your_username',
#     'password': 'your_password'
# }

# يمكن ترك TABLES إذا كنت ستستخدم الأسماء من هنا لاحقًا، لكن استعلاماتك الحالية تستخدم الأسماء مباشرة
TABLES = {
    'problems': 'problem',
    'solutions': 'proposed_solution',
    'implementations': 'implementation_plan',
    'kpis': 'kpi_measurement',
    'lessons': 'lesson_learned',
    'problem_understanding': 'problem_understanding',
    'cause_analysis': 'cause_analysis',
    'chosen_solution': 'chosen_solution',
    'potential_root_cause': 'potential_root_cause',
    # ... أضف باقي الجداول إذا احتجت إليها هنا
}