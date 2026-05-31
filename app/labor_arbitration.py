"""劳动仲裁申请书字段抽取、缺失检查与模板渲染。"""

from __future__ import annotations

import calendar
import re
from datetime import datetime
from typing import Any


DOC_TYPE = "labor_arbitration_application"
DOC_TYPE_ALIASES = {"labor_arbitration", DOC_TYPE}

FIELD_LABELS = {
    "applicant_name": "申请人姓名",
    "applicant_gender": "申请人性别",
    "applicant_birth_date": "申请人出生日期",
    "applicant_phone": "申请人联系方式",
    "applicant_id_number": "申请人身份证号",
    "applicant_address": "申请人住所",
    "applicant_mailing_address": "申请人通讯地址",
    "respondent_name": "被申请人公司名称",
    "respondent_address": "公司地址",
    "respondent_mailing_address": "被申请人通讯地址",
    "respondent_legal_rep": "法定代表人或主要负责人",
    "legal_representative": "法定代表人或主要负责人",
    "respondent_contact_person": "联络人及职务",
    "contact_person_and_title": "联络人及职务",
    "respondent_phone": "被申请人联系电话",
    "employment_start_date": "入职时间",
    "employment_end_date": "离职/辞退时间",
    "job_position": "岗位及职务",
    "position": "岗位及职务",
    "contract_last_period": "最后一期劳动合同期限",
    "last_contract_period": "最后一期劳动合同期限",
    "work_location": "工作地点",
    "work_hours": "工作时间",
    "working_hours": "工作时间",
    "requires_attendance": "是否需要考勤",
    "attendance_method": "考勤方式",
    "salary_payment_method": "工资发放方式",
    "initial_salary": "入职时工资标准",
    "salary_adjustment": "工资标准调整情况",
    "current_employment_status": "现是否在职",
    "is_currently_employed": "现是否在职",
    "average_salary_12_months": "离职前 12 个月的月平均工资",
    "average_salary_last_12_months": "离职前 12 个月的月平均工资",
    "monthly_salary": "月工资",
    "has_written_contract": "是否签订书面劳动合同",
    "has_social_insurance": "是否缴纳社保",
    "termination_reason": "离职/辞退原因",
    "arbitration_claims": "仲裁请求",
    "claim_calculation_formula": "仲裁请求计算公式",
    "claim_calculation": "仲裁请求计算公式",
    "facts": "事实与理由要点",
    "facts_and_reasons": "事实与理由",
    "evidence_list": "证据材料",
    "arbitration_commission": "仲裁委员会名称",
    "other_facts": "其他需要说明的事实和理由",
}

LABOR_FIELDS = {
    "applicant_name": "",
    "applicant_gender": "",
    "applicant_birth_date": "",
    "applicant_phone": "",
    "applicant_id_number": "",
    "applicant_address": "",
    "applicant_mailing_address": "",
    "respondent_name": "",
    "respondent_address": "",
    "respondent_mailing_address": "",
    "respondent_legal_rep": "",
    "legal_representative": "",
    "respondent_contact_person": "",
    "contact_person_and_title": "",
    "respondent_phone": "",
    "employment_start_date": "",
    "employment_end_date": "",
    "job_position": "",
    "position": "",
    "contract_last_period": "",
    "last_contract_period": "",
    "work_location": "",
    "work_hours": "",
    "working_hours": "",
    "requires_attendance": "",
    "attendance_method": "",
    "salary_payment_method": "",
    "initial_salary": "",
    "salary_adjustment": "",
    "current_employment_status": "",
    "is_currently_employed": "",
    "average_salary_12_months": "",
    "average_salary_last_12_months": "",
    "monthly_salary": "",
    "has_written_contract": "",
    "has_social_insurance": "",
    "termination_reason": "",
    "arbitration_claims": [],
    "claim_calculation_formula": [],
    "claim_calculation": "",
    "facts": [],
    "facts_and_reasons": "",
    "evidence_list": [],
    "arbitration_commission": "",
    "other_facts": "",
}

REQUIRED_FIELDS = [
    "applicant_name",
    "respondent_name",
    "employment_start_date",
    "employment_end_date",
    "monthly_salary",
]


def canonical_doc_type(doc_type: str) -> str:
    return DOC_TYPE if doc_type in DOC_TYPE_ALIASES else doc_type


def _clean(value: str) -> str:
    return (value or "").strip().strip("，,。；;：: ")


def _first_match(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean(match.group(1))
    return ""


def _normalize_date(value: str) -> str:
    value = _clean(value)
    if not value:
        return ""
    match = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?", value)
    if match:
        return f"{match.group(1)}年{int(match.group(2))}月{int(match.group(3))}日"
    match = re.search(r"(\d{4})[年/-](\d{1,2})月?", value)
    if match:
        return f"{match.group(1)}年{int(match.group(2))}月"
    return value


def _parse_chinese_date(value: str) -> datetime | None:
    value = _clean(value)
    match = re.search(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?", value)
    if not match:
        match = re.search(r"(\d{4})[/-](\d{1,2})(?:[/-](\d{1,2}))?", value)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or 1)
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def _add_months(dt: datetime, months: int) -> datetime:
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def _format_date(dt: datetime | None) -> str:
    if not dt:
        return ""
    return f"{dt.year}年{dt.month}月{dt.day}日"


def _money_number(value: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _format_money(value: float) -> str:
    if value.is_integer():
        return f"{int(value)}元"
    return f"{value:.2f}元"


def _service_months(start: str, end: str) -> int | None:
    start_dt = _parse_chinese_date(start)
    end_dt = _parse_chinese_date(end)
    if not start_dt or not end_dt or end_dt < start_dt:
        return None
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    if end_dt.day >= start_dt.day:
        months += 1
    return max(months, 1)


def _compensation_months(start: str, end: str) -> float | None:
    months = _service_months(start, end)
    if months is None:
        return None
    full_years, remainder = divmod(months, 12)
    if remainder == 0:
        return float(full_years)
    if remainder <= 6:
        return full_years + 0.5
    return full_years + 1.0


def empty_fields() -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, list) else v) for k, v in LABOR_FIELDS.items()}


def _sync_alias_fields(fields: dict[str, Any]) -> None:
    alias_pairs = [
        ("respondent_legal_rep", "legal_representative"),
        ("respondent_contact_person", "contact_person_and_title"),
        ("job_position", "position"),
        ("contract_last_period", "last_contract_period"),
        ("work_hours", "working_hours"),
        ("current_employment_status", "is_currently_employed"),
        ("average_salary_12_months", "average_salary_last_12_months"),
    ]
    for old_key, new_key in alias_pairs:
        if fields.get(old_key) and not fields.get(new_key):
            fields[new_key] = fields[old_key]
        elif fields.get(new_key) and not fields.get(old_key):
            fields[old_key] = fields[new_key]
    if fields.get("claim_calculation_formula") and not fields.get("claim_calculation"):
        fields["claim_calculation"] = "\n".join(fields["claim_calculation_formula"])
    elif fields.get("claim_calculation") and not fields.get("claim_calculation_formula"):
        fields["claim_calculation_formula"] = _merge_list("", [fields["claim_calculation"]])
    if fields.get("facts") and not fields.get("facts_and_reasons"):
        fields["facts_and_reasons"] = "\n".join(fields["facts"])
    elif fields.get("facts_and_reasons") and not fields.get("facts"):
        fields["facts"] = _merge_list("", [fields["facts_and_reasons"]])


def extract_labor_arbitration_fields(text: str, existing: dict | None = None) -> dict[str, Any]:
    fields = empty_fields()
    if existing:
        for key, value in existing.items():
            if key in fields and value:
                fields[key] = value
        _sync_alias_fields(fields)

    text = text or ""

    applicant_name = _first_match([
        r"申请人(?:是|为)?([一-龥]{2,4})(?:，|,|。|电话|联系方式|身份证|$)",
        r"我是([一-龥]{2,4})(?:，|,|。|电话|联系方式|身份证|$)",
        r"我叫([一-龥]{2,4})(?:，|,|。|电话|联系方式|身份证|$)",
    ], text)
    if applicant_name:
        fields["applicant_name"] = applicant_name

    gender = _first_match([
        r"性别(?:是|为)?(男|女)",
        r"(男|女)(?:，|,|。|出生|身份证)",
    ], text)
    if gender:
        fields["applicant_gender"] = gender

    respondent_name = _first_match([
        r"被申请人(?:是|为)?([^，。,；;\n]+?(?:公司|有限公司|集团|厂|店|中心))",
        r"(?:公司|单位)(?:是|为|名称是)?([^，。,；;\n]+?(?:公司|有限公司|集团|厂|店|中心))",
    ], text)
    if respondent_name:
        fields["respondent_name"] = respondent_name

    respondent_address = _first_match([
        r"(?:公司)?地址(?:是|为|在)?([^，。,；;\n]+)",
        r"住所地(?:是|为)?([^，。,；;\n]+)",
    ], text)
    if respondent_address:
        fields["respondent_address"] = respondent_address
        fields["respondent_mailing_address"] = fields.get("respondent_mailing_address") or respondent_address

    applicant_address = _first_match([
        r"(?:申请人)?住所(?:是|为|在)?([^，。,；;\n]+)",
        r"(?:现住址|住址|居住地址)(?:是|为|在)?([^，。,；;\n]+)",
    ], text)
    if applicant_address:
        fields["applicant_address"] = applicant_address
        fields["applicant_mailing_address"] = fields.get("applicant_mailing_address") or applicant_address

    applicant_mailing_address = _first_match([
        r"(?:申请人)?通讯地址(?:是|为|在)?([^，。,；;\n]+)",
    ], text)
    if applicant_mailing_address:
        fields["applicant_mailing_address"] = applicant_mailing_address

    respondent_mailing_address = _first_match([
        r"(?:被申请人|公司|单位)?通讯地址(?:是|为|在)?([^，。,；;\n]+)",
    ], text)
    if respondent_mailing_address:
        fields["respondent_mailing_address"] = respondent_mailing_address

    legal_rep = _first_match([
        r"(?:法定代表人|主要负责人)(?:是|为)?([一-龥]{2,4})",
    ], text)
    if legal_rep:
        fields["respondent_legal_rep"] = legal_rep
        fields["legal_representative"] = legal_rep

    contact_person = _first_match([
        r"(?:联络人|联系人)(?:及职务)?(?:是|为)?([^，。,；;\n]+)",
    ], text)
    if contact_person:
        fields["respondent_contact_person"] = contact_person
        fields["contact_person_and_title"] = contact_person

    respondent_phone = _first_match([
        r"(?:公司|单位|被申请人)(?:电话|联系电话)(?:是|为)?\s*([0-9\-]{7,20})",
    ], text)
    if respondent_phone:
        fields["respondent_phone"] = respondent_phone

    phone = _first_match([r"(1[3-9]\d{9})"], text)
    if phone:
        fields["applicant_phone"] = phone

    id_number = _first_match([r"([1-9]\d{5}(?:18|19|20)\d{2}\d{2}\d{2}\d{3}[\dXx])"], text)
    if id_number:
        fields["applicant_id_number"] = id_number
        fields["applicant_birth_date"] = fields.get("applicant_birth_date") or f"{id_number[6:10]}年{int(id_number[10:12])}月{int(id_number[12:14])}日"
        fields["applicant_gender"] = fields.get("applicant_gender") or ("男" if int(id_number[16]) % 2 else "女")

    birth_date = _first_match([
        r"出生日期(?:是|为)?(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"出生于(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
    ], text)
    if birth_date:
        fields["applicant_birth_date"] = _normalize_date(birth_date)

    start_date = _first_match([
        r"(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)入职",
        r"(\d{4}[年/-]\d{1,2}月?)入职",
        r"入职(?:时间)?(?:是|为)?(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)",
        r"入职(?:时间)?(?:是|为)?(\d{4}[年/-]\d{1,2}月?)",
    ], text)
    if start_date:
        fields["employment_start_date"] = _normalize_date(start_date)

    end_date = _first_match([
        r"(\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)(?:被辞退|离职|解除|开除)",
        r"(\d{4}[年/-]\d{1,2}月?)(?:被辞退|离职|解除|开除)",
    ], text)
    if end_date:
        fields["employment_end_date"] = _normalize_date(end_date)

    salary = _first_match([
        r"(?:月工资|工资|每月工资|每个月工资)(?:是|为)?\s*([0-9]+(?:\.[0-9]+)?\s*元?)",
        r"([0-9]+(?:\.[0-9]+)?\s*元)(?:/月|每月|一个月|月工资)",
    ], text)
    if salary:
        fields["monthly_salary"] = salary if "元" in salary else f"{salary}元"
        fields["average_salary_12_months"] = fields.get("average_salary_12_months") or fields["monthly_salary"]
        fields["average_salary_last_12_months"] = fields.get("average_salary_last_12_months") or fields["monthly_salary"]
        fields["initial_salary"] = fields.get("initial_salary") or fields["monthly_salary"]

    avg_salary = _first_match([
        r"(?:离职前\s*12\s*个月的?)?月平均工资(?:是|为)?\s*([0-9]+(?:\.[0-9]+)?\s*元?)",
        r"离职前\s*12\s*个月.*?平均(?:工资)?(?:是|为)?\s*([0-9]+(?:\.[0-9]+)?\s*元?)",
    ], text)
    if avg_salary:
        fields["average_salary_12_months"] = avg_salary if "元" in avg_salary else f"{avg_salary}元"
        fields["average_salary_last_12_months"] = fields["average_salary_12_months"]

    initial_salary = _first_match([
        r"入职时(?:工资标准|工资)(?:是|为)?\s*([0-9]+(?:\.[0-9]+)?\s*元?)",
    ], text)
    if initial_salary:
        fields["initial_salary"] = initial_salary if "元" in initial_salary else f"{initial_salary}元"

    job_position = _first_match([
        r"(?:岗位|职位|职务)(?:是|为)?([^，。,；;\n]+)",
        r"担任([^，。,；;\n]+?)(?:岗位|职务|职位)",
    ], text)
    if job_position:
        fields["job_position"] = job_position
        fields["position"] = job_position

    work_location = _first_match([
        r"工作地点(?:是|为|在)?([^，。,；;\n]+)",
    ], text)
    if work_location:
        fields["work_location"] = work_location

    work_hours = _first_match([
        r"工作时间(?:是|为)?([^，。,；;\n]+)",
        r"(?:每天|每日)(工作[^，。,；;\n]+)",
    ], text)
    if work_hours:
        fields["work_hours"] = work_hours
        fields["working_hours"] = work_hours

    attendance_method = _first_match([
        r"考勤方式(?:是|为)?([^，。,；;\n]+)",
        r"通过([^，。,；;\n]+?)(?:考勤|打卡)",
    ], text)
    if attendance_method:
        fields["attendance_method"] = attendance_method
        fields["requires_attendance"] = fields.get("requires_attendance") or "是"
    elif re.search(r"考勤|打卡", text):
        fields["requires_attendance"] = "是"

    salary_payment_method = _first_match([
        r"工资(?:发放方式|支付方式)(?:是|为)?([^，。,；;\n]+)",
        r"(?:通过|以)(银行转账|现金|微信|支付宝)(?:方式)?发(?:放|工资)",
    ], text)
    if salary_payment_method:
        fields["salary_payment_method"] = salary_payment_method

    salary_adjustment = _first_match([
        r"工资标准调整情况(?:是|为)?([^。；;\n]+)",
        r"工资(?:调整|涨薪|降薪)(?:情况)?(?:是|为)?([^。；;\n]+)",
    ], text)
    if salary_adjustment:
        fields["salary_adjustment"] = salary_adjustment

    if re.search(r"没有(?:和我)?签(?:订)?(?:书面)?劳动合同|未签(?:订)?(?:书面)?劳动合同|没签劳动合同", text):
        fields["has_written_contract"] = "否"
    elif re.search(r"签(?:订)?了(?:书面)?劳动合同|有劳动合同", text):
        fields["has_written_contract"] = "是"

    contract_last_period = _first_match([
        r"最后一期劳动合同期限(?:是|为)?([^。；;\n]+)",
        r"劳动合同期限(?:是|为)?([^。；;\n]+)",
    ], text)
    if contract_last_period:
        fields["contract_last_period"] = contract_last_period
        fields["last_contract_period"] = contract_last_period

    if re.search(r"没有(?:给我)?(?:缴纳|交)社保|未缴纳社保|没(?:有)?社保", text):
        fields["has_social_insurance"] = "否"
    elif re.search(r"(?:缴纳|交)了?社保|有社保", text):
        fields["has_social_insurance"] = "是"

    if re.search(r"辞退|不用来了|解雇|开除|解除劳动关系", text):
        fields["termination_reason"] = "被申请人口头或单方解除劳动关系，具体原因待补充"
        fields["current_employment_status"] = "否"
        fields["is_currently_employed"] = "否"
    elif "离职" in text:
        fields["termination_reason"] = "离职原因待补充"
        fields["current_employment_status"] = "否"
        fields["is_currently_employed"] = "否"
    elif re.search(r"仍在职|还在职|目前在职|现仍工作", text):
        fields["current_employment_status"] = "是"
        fields["is_currently_employed"] = "是"

    commission = _first_match([
        r"((?:[\u4e00-\u9fff]{2,20})(?:劳动人事争议调解仲裁院|劳动人事争议仲裁委员会|劳动争议仲裁委员会))",
    ], text)
    if commission:
        fields["arbitration_commission"] = commission
    elif not fields.get("arbitration_commission"):
        fields["arbitration_commission"] = "待补充劳动人事争议仲裁委员会"

    facts = []
    if fields["employment_start_date"]:
        facts.append(f"申请人于{fields['employment_start_date']}入职被申请人处。")
    if fields["employment_end_date"]:
        facts.append(f"双方劳动关系于{fields['employment_end_date']}发生解除或终止。")
    if fields["monthly_salary"]:
        facts.append(f"申请人月工资标准为{fields['monthly_salary']}。")
    if fields["has_written_contract"] == "否":
        facts.append("被申请人未与申请人签订书面劳动合同。")
    if fields["has_social_insurance"] == "否":
        facts.append("被申请人未依法为申请人缴纳社会保险。")
    if fields["termination_reason"]:
        facts.append(fields["termination_reason"])
    fields["facts"] = _merge_list(fields.get("facts", []), facts)

    claims = infer_labor_claims(fields, text)
    fields["arbitration_claims"] = _merge_list(fields.get("arbitration_claims", []), claims)
    formulas = infer_claim_calculation_formula(fields, text)
    fields["claim_calculation_formula"] = _merge_list(fields.get("claim_calculation_formula", []), formulas)
    fields["claim_calculation"] = "\n".join(fields["claim_calculation_formula"])
    evidence = infer_evidence_list(fields, text)
    fields["evidence_list"] = _merge_list(fields.get("evidence_list", []), evidence)
    fields["facts_and_reasons"] = "\n".join(fields.get("facts", []))
    _sync_alias_fields(fields)

    return fields


def _merge_list(old: Any, new: list[str]) -> list[str]:
    items = []
    if isinstance(old, list):
        items.extend(str(x) for x in old if str(x).strip())
    elif isinstance(old, str) and old.strip():
        items.extend(x.strip() for x in re.split(r"\n|；|;", old) if x.strip())
    for item in new:
        if item and item not in items:
            items.append(item)
    return items


def infer_labor_claims(fields: dict[str, Any], text: str) -> list[str]:
    claims = []
    salary = fields.get("average_salary_12_months") or fields.get("monthly_salary")
    start = fields.get("employment_start_date", "")
    end = fields.get("employment_end_date", "")
    salary_num = _money_number(salary)

    if fields.get("has_written_contract") == "否":
        if salary_num:
            claims.append(
                f"请求裁决被申请人支付未签订书面劳动合同二倍工资差额，暂按{salary}×11个月计算为{_format_money(salary_num * 11)}，具体期间和金额以仲裁庭核算为准。"
            )
        else:
            claims.append("请求裁决被申请人支付未签订书面劳动合同二倍工资差额，具体金额待补充工资标准后计算。")
    if re.search(r"辞退|不用来了|解雇|开除|解除劳动关系|不给赔偿|没有给.*赔偿", text):
        compensation_months = _compensation_months(start, end)
        if salary_num and compensation_months:
            amount = salary_num * compensation_months * 2
            claims.append(
                f"如仲裁庭认定被申请人违法解除劳动关系，请求裁决被申请人支付违法解除赔偿金，暂按{salary}×{compensation_months:g}个月×2计算为{_format_money(amount)}，最终以仲裁认定为准。"
            )
        else:
            claims.append("如仲裁庭认定被申请人违法解除劳动关系，请求裁决被申请人支付违法解除赔偿金；金额需补充入职时间、离职时间和月平均工资后计算。")
    if re.search(r"拖欠工资|欠工资|工资未发|没发工资|克扣工资", text):
        claims.append("请求裁决被申请人支付拖欠或克扣的工资，具体期间和金额待补充欠付月份、已发金额后计算。")
    if fields.get("has_social_insurance") == "否":
        claims.append("请求确认被申请人未依法缴纳社会保险，并依法承担相应补缴情形下的责任。")
    if not claims:
        claims.append("请求依法维护申请人的劳动权益，具体仲裁请求待补充。")
    return claims


def infer_claim_calculation_formula(fields: dict[str, Any], text: str) -> list[str]:
    formulas = []
    salary = fields.get("average_salary_12_months") or fields.get("monthly_salary")
    salary_num = _money_number(salary)
    start = fields.get("employment_start_date", "")
    end = fields.get("employment_end_date", "")

    if fields.get("has_written_contract") == "否":
        if salary_num:
            formulas.append(
                f"未签订书面劳动合同二倍工资差额：{salary}×11个月={_format_money(salary_num * 11)}；实际起止期间以入职次月、满一年前一日及仲裁时效认定为准。"
            )
        else:
            formulas.append("未签订书面劳动合同二倍工资差额：月工资标准×可主张月份；需补充工资标准和可主张期间。")

    if re.search(r"辞退|不用来了|解雇|开除|解除劳动关系|不给赔偿|没有给.*赔偿", text):
        compensation_months = _compensation_months(start, end)
        if salary_num and compensation_months:
            formulas.append(
                f"违法解除赔偿金：离职前12个月月平均工资{salary}×经济补偿年限{compensation_months:g}个月×2={_format_money(salary_num * compensation_months * 2)}。"
            )
        else:
            formulas.append("违法解除赔偿金：离职前12个月月平均工资×经济补偿年限×2；需补充入职时间、离职时间和月平均工资。")

    if re.search(r"拖欠工资|欠工资|工资未发|没发工资|克扣工资", text):
        formulas.append("拖欠工资：欠付月份应发工资合计-已发工资合计；需补充欠付期间、工资标准和已支付金额。")

    if fields.get("has_social_insurance") == "否":
        formulas.append("社会保险相关请求：补缴期间和基数以社保经办机构及仲裁庭认定为准。")

    return formulas or ["待补充具体请求对应的计算公式。"]


def infer_evidence_list(fields: dict[str, Any], text: str) -> list[str]:
    evidence = [
        "工资流水、银行转账记录或工资条，证明工资标准和工资支付情况。",
        "工作证、考勤记录、聊天记录、工牌、入职材料等，证明劳动关系和工作期间。",
    ]
    if fields.get("has_written_contract") == "否":
        evidence.append("未签订书面劳动合同的相关说明及入职、工作沟通记录。")
    if re.search(r"辞退|不用来了|解雇|开除|解除劳动关系", text):
        evidence.append("辞退通知、聊天记录、录音或其他解除劳动关系证据。")
    if fields.get("has_social_insurance") == "否":
        evidence.append("社保缴费记录或社保查询截图，证明社保缴纳情况。")
    return evidence


def missing_required_fields(fields: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_FIELDS if not fields.get(key)]


def missing_fields_message(missing: list[str]) -> str:
    lines = ["生成劳动人事争议仲裁申请书前，还需要补充以下信息：", ""]
    for i, key in enumerate(missing, 1):
        lines.append(f"{i}. {FIELD_LABELS.get(key, key)}")
    lines.extend(["", "请补充后，我将继续生成文书。"])
    return "\n".join(lines)


def _numbered(items: list[str]) -> str:
    if not items:
        return "1. 待补充"
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _numbered_table_value(items: list[str]) -> str:
    if not items:
        return "待补充"
    return "<br>".join(f"{i}. {_escape_table(item)}" for i, item in enumerate(items, 1))


def _value(fields: dict[str, Any], key: str) -> str:
    return str(fields.get(key) or "待补充")


def _escape_table(value: Any) -> str:
    text = str(value or "待补充")
    return text.replace("|", "\\|").replace("\n", "<br>")


def _row(label: str, value: Any) -> str:
    return f"| {label} | {_escape_table(value)} |"


def render_labor_arbitration_application(fields: dict[str, Any]) -> str:
    other_facts = list(fields.get("facts") or [])
    if fields.get("has_written_contract") == "否":
        note = "未签订书面劳动合同相关责任及可主张期间以仲裁庭查明为准。"
        if note not in other_facts:
            other_facts.append(note)
    if fields.get("has_social_insurance") == "否":
        note = "未依法缴纳社会保险的相关处理以社保经办机构及仲裁庭认定为准。"
        if note not in other_facts:
            other_facts.append(note)
    if re.search(r"违法解除|单方解除|口头", fields.get("termination_reason", "")):
        note = "关于解除劳动关系是否合法及赔偿/补偿金额，应结合解除原因、通知方式、证据材料和仲裁庭认定确定。"
        if note not in other_facts:
            other_facts.append(note)

    document = f"""# 劳动人事争议仲裁申请书

致：{_value(fields, "arbitration_commission")}

## 一、申请人信息

| 项目 | 内容 |
| --- | --- |
{_row("姓名", _value(fields, "applicant_name"))}
{_row("性别", _value(fields, "applicant_gender"))}
{_row("出生日期", _value(fields, "applicant_birth_date"))}
{_row("公民身份号码", _value(fields, "applicant_id_number"))}
{_row("联系电话", _value(fields, "applicant_phone"))}
{_row("住所", _value(fields, "applicant_address"))}
{_row("通讯地址", _value(fields, "applicant_mailing_address"))}

## 二、被申请人信息

| 项目 | 内容 |
| --- | --- |
{_row("名称", _value(fields, "respondent_name"))}
{_row("住所", _value(fields, "respondent_address"))}
{_row("通讯地址", _value(fields, "respondent_mailing_address"))}
{_row("法定代表人或主要负责人", _value(fields, "respondent_legal_rep"))}
{_row("联络人及职务", _value(fields, "respondent_contact_person"))}
{_row("联系电话", _value(fields, "respondent_phone"))}

## 三、仲裁请求

| 项目 | 内容 |
| --- | --- |
{_row("请求 1", (fields.get("arbitration_claims") or ["待补充"])[0])}
{_row("请求 2", (fields.get("arbitration_claims") or ["", "待补充"])[1] if len(fields.get("arbitration_claims") or []) > 1 else "待补充")}
{_row("请求 3", (fields.get("arbitration_claims") or ["", "", "待补充"])[2] if len(fields.get("arbitration_claims") or []) > 2 else "待补充")}
{_row("仲裁请求计算公式", _numbered_table_value(fields.get("claim_calculation_formula") or []))}

## 四、基本事实和理由

| 项目 | 内容 |
| --- | --- |
{_row("入职时间", _value(fields, "employment_start_date"))}
{_row("岗位及职务", _value(fields, "job_position"))}
{_row("有无签订劳动合同", _value(fields, "has_written_contract"))}
{_row("最后一期劳动合同期限", _value(fields, "contract_last_period"))}
{_row("工作地点", _value(fields, "work_location"))}
{_row("工作时间", _value(fields, "work_hours"))}
{_row("是否需要考勤", _value(fields, "requires_attendance"))}
{_row("考勤方式", _value(fields, "attendance_method"))}
{_row("工资发放方式", _value(fields, "salary_payment_method"))}
{_row("入职时工资标准", _value(fields, "initial_salary"))}
{_row("工资标准调整情况", _value(fields, "salary_adjustment"))}
{_row("现是否在职", _value(fields, "current_employment_status"))}
{_row("离职时间", _value(fields, "employment_end_date"))}
{_row("离职原因", _value(fields, "termination_reason"))}
{_row("离职前 12 个月的月平均工资", _value(fields, "average_salary_12_months"))}
{_row("其他需要说明的事实和理由", _numbered_table_value(_merge_list(fields.get("other_facts", ""), other_facts)))}

## 五、证据目录

{_numbered(fields.get("evidence_list") or [])}

此致

{_value(fields, "arbitration_commission")}

免责声明：本文书由系统根据用户提供的信息辅助生成，仅供参考，不构成正式法律意见。提交前建议咨询专业律师或当地劳动仲裁机构。

申请人：{_value(fields, "applicant_name")}

提交日期：{datetime.now().strftime("%Y年%m月%d日")}
"""
    return document


def build_labor_document_result(text: str, existing_fields: dict | None = None) -> dict[str, Any]:
    fields = extract_labor_arbitration_fields(text, existing=existing_fields)
    missing = missing_required_fields(fields)
    if missing:
        return {
            "type": "document_generation_result",
            "doc_type": DOC_TYPE,
            "status": "missing_fields",
            "missing_fields": missing,
            "extracted_fields": fields,
            "form_fields": fields,
            "display_modes": ["form", "preview"],
            "message": missing_fields_message(missing),
        }
    document = render_labor_arbitration_application(fields)
    return {
        "type": "document_generation_result",
        "doc_type": DOC_TYPE,
        "status": "success",
        "missing_fields": [],
        "extracted_fields": fields,
        "form_fields": fields,
        "display_modes": ["form", "preview"],
        "document_markdown": document,
        "warnings": ["本文书仅供参考，提交前建议咨询专业律师。"],
    }
