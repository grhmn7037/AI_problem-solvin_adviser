# src/utils/feature_engineering_utils.py
import pandas as pd
import numpy as np
import re


def parse_cost_value(cost_str: str) -> float:
    """
    يحلل قيمة التكلفة من نص ويحولها إلى رقم float.
    يعالج الأرقام، النطاقات، والكلمات الوصفية (عالي، متوسط، منخفض).
    """
    if pd.isna(cost_str) or not isinstance(cost_str, str) or cost_str.strip() == '':
        return np.nan

    cost_str = str(cost_str).lower()  # توحيد حالة الأحرف
    numeric_value = np.nan

    numbers_found = re.findall(r'\d+\.?\d*', cost_str)

    if numbers_found:
        if len(numbers_found) == 1:
            numeric_value = float(numbers_found[0])
        elif len(numbers_found) > 1 and ('-' in cost_str or 'الى' in cost_str or 'إلى' in cost_str):
            try:
                numeric_value = (float(numbers_found[0]) + float(numbers_found[-1])) / 2
            except ValueError:
                numeric_value = float(numbers_found[0])  # إذا فشل، خذ الرقم الأول
        else:  # إذا كان هناك عدة أرقام غير مرتبطة بنطاق واضح، خذ الأول كافتراض
            numeric_value = float(numbers_found[0])
    else:
        if "عالي" in cost_str or "مرتفع" in cost_str:
            numeric_value = 10000.0
        elif "متوسط" in cost_str:  # يمكنك إضافة "متوسط جدا" هنا إذا أردت
            if "جدا" in cost_str and "متوسط" in cost_str:  # معالجة "متوسط جدا"
                numeric_value = 7500.0  # أو أي قيمة تراها مناسبة
            elif "متوسط" in cost_str:
                numeric_value = 5000.0
        elif "منخفض" in cost_str:
            numeric_value = 1000.0

    return numeric_value


def parse_time_to_implement(time_str: str) -> float:
    """
    يحلل وقت التنفيذ من نص ويحوله إلى عدد الأيام (float).
    يعالج "فوري"، الأرقام مع وحدات (شهر، أسبوع، يوم، ساعة، دقيقة)، والنطاقات.
    """
    if pd.isna(time_str) or not isinstance(time_str, str) or time_str.strip() == '':
        return np.nan

    time_str = str(time_str).lower()
    days = np.nan

    if "فوري" in time_str:
        return 0.0

    numbers_found = re.findall(r'\d+\.?\d*', time_str)
    avg_value = None
    if numbers_found:
        value1 = float(numbers_found[0])
        avg_value = value1
        if len(numbers_found) > 1 and ('-' in time_str or 'الى' in time_str or 'إلى' in time_str):
            value2 = float(numbers_found[-1])
            avg_value = (value1 + value2) / 2

    if avg_value is None:  # إذا لم يتم العثور على أرقام بعد "فوري"
        return np.nan

    if "شهر" in time_str or "اشهر" in time_str or "أشهر" in time_str:
        days = avg_value * 30
    elif "اسبوع" in time_str or "أسبوع" in time_str or "اسابيع" in time_str or "أسابيع" in time_str:
        days = avg_value * 7
    elif "يوم" in time_str or "ايام" in time_str or "أيام" in time_str:
        days = avg_value
    elif "ساعه" in time_str or "ساعة" in time_str or "ساعات" in time_str:
        days = avg_value / 24
    elif "دقيقه" in time_str or "دقيقة" in time_str or "دقائق" in time_str:
        days = avg_value / (24 * 60)
    elif numbers_found:  # إذا وجدت أرقام ولكن لم يتم تحديد وحدة واضحة
        # يمكنك أن تقرر هنا، هل تفترض أنها أيام؟ أم تتركها NaN؟
        # الافتراض بأنها أيام قد يكون خاطئًا. من الأفضل تركها NaN إذا لم تكن الوحدة واضحة.
        # print(f"تحذير: تم العثور على رقم '{avg_value}' في '{time_str}' بدون وحدة زمنية واضحة. سيتم إرجاع NaN.")
        return np.nan
    else:  # لا أرقام ولا كلمة "فوري" ولا وحدات زمنية واضحة
        return np.nan

    return days


if __name__ == '__main__':
    # اختبارات بسيطة للدوال
    print("--- اختبار دوال feature_engineering_utils ---")
    print(f"'متوسط جدا': {parse_cost_value('متوسط جدا')} (Expected: 7500.0 or similar)")
    print(f"'5000-7000 دولار': {parse_cost_value('5000-7000 دولار')} (Expected: 6000.0)")
    print(f"'عالي': {parse_cost_value('عالي')} (Expected: 10000.0)")
    print(f"'150 ريال': {parse_cost_value('150 ريال')} (Expected: 150.0)")
    print(f"None: {parse_cost_value(None)} (Expected: nan)")
    print(f"'': {parse_cost_value('')} (Expected: nan)")

    print(f"'3 اسابيع': {parse_time_to_implement('3 اسابيع')} (Expected: 21.0)")
    print(f"'حوالي 2-4 أشهر': {parse_time_to_implement('حوالي 2-4 أشهر')} (Expected: 90.0)")
    print(f"'فوري': {parse_time_to_implement('فوري')} (Expected: 0.0)")
    print(f"'12 ساعة': {parse_time_to_implement('12 ساعة')} (Expected: 0.5)")
    print(f"'نص بدون رقم أو وحدة': {parse_time_to_implement('نص بدون رقم أو وحدة')} (Expected: nan)")
    print(f"'30': {parse_time_to_implement('30')} (Expected: nan, because no unit)")  # أو 30.0 إذا أردت افتراض أيام