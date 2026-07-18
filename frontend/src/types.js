/**
 * @typedef {Object} RagSubstep
 * @property {string} step
 * @property {string=} label
 * @property {"started"|"done"|"failed"|"pending"|"skipped"=} status
 * @property {number=} duration_ms
 * @property {number=} elapsed_ms
 * @property {number=} input_count
 * @property {number=} output_count
 * @property {string=} message
 */

/**
 * @typedef {Object} SourceItem
 * @property {string=} source_id
 * @property {string=} source_type
 * @property {string=} law_name
 * @property {string=} article_no
 * @property {string=} text
 * @property {string=} content
 * @property {string|number=} confidence
 * @property {string=} source_level
 * @property {string|number=} authority_weight
 */

/**
 * @typedef {Object} CaseAnalysisClaim
 * @property {string=} claim
 * @property {string=} domain
 * @property {Array=} basis
 * @property {string[]=} supporting_facts
 * @property {string[]=} missing_facts
 * @property {string[]=} risks
 * @property {string[]=} suggested_evidence
 */

/**
 * @typedef {Object} CaseAnalysisResult
 * @property {string=} case_summary
 * @property {string=} case_type
 * @property {"low"|"medium"|"high"|"unknown"=} risk_level
 * @property {string[]=} legal_relationships
 * @property {CaseAnalysisClaim[]=} claims
 * @property {string[]=} evidence_checklist
 * @property {string[]=} next_actions
 * @property {string[]=} missing_information
 * @property {string[]=} document_suggestions
 */

/**
 * @typedef {Object} CaseAction
 * @property {string=} id
 * @property {string} type
 * @property {string=} doc_type
 * @property {string=} label
 * @property {string=} source
 * @property {string=} case_analysis_id
 */

/**
 * @typedef {Object} CaseState
 * @property {string=} case_analysis_id
 * @property {CaseAnalysisResult=} analysis_result
 * @property {CaseAction[]=} available_actions
 */

/**
 * @typedef {Object} DocumentResult
 * @property {string=} status
 * @property {string=} document_type
 * @property {Object=} extracted_fields
 * @property {Object=} form_fields
 * @property {Array=} missing_fields
 * @property {Array=} missing_field_details
 * @property {string[]=} clarification_questions
 * @property {string=} document_markdown
 * @property {boolean=} can_export
 * @property {Object=} field_status
 * @property {Array=} field_slots
 */

/**
 * @typedef {Object} DocumentStreamParams
 * @property {CaseState|null=} case_state
 * @property {string=} sessionId
 * @property {string=} extra_info
 * @property {string=} action
 * @property {string=} source
 * @property {string=} case_analysis_id
 * @property {string=} doc_type
 */

export {}
