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
 * @property {CaseAction[]=} available_actions
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
