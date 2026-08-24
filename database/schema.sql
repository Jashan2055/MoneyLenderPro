CREATE DATABASE IF NOT EXISTS micro_lending;

USE micro_lending;

DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS customers;


CREATE TABLE customers (
    customer_id BIGINT PRIMARY KEY,

    emp_title VARCHAR(255),
    emp_length VARCHAR(50),
    home_ownership VARCHAR(50),

    annual_income DECIMAL(15,2),

    verification_status VARCHAR(50),
    addr_state VARCHAR(10),

    INDEX idx_customer_state (addr_state)
);


CREATE TABLE loans (
    loan_id BIGINT PRIMARY KEY,

    customer_id BIGINT NOT NULL,

    loan_amount DECIMAL(15,2),
    funded_amount DECIMAL(15,2),
    investor_funds DECIMAL(15,2),

    term VARCHAR(20),
    interest_rate DECIMAL(8,4),
    installment DECIMAL(15,2),

    grade VARCHAR(5),
    sub_grade VARCHAR(10),

    issue_d VARCHAR(20),
    loan_status VARCHAR(100),

    pymnt_plan VARCHAR(10),
    purpose VARCHAR(100),
    title VARCHAR(255),
    zip_code VARCHAR(10),

    dti DECIMAL(10,4),

    delinq_2yrs INT,
    earliest_cr_line VARCHAR(20),
    inq_last_6mths INT,
    mths_since_last_delinq INT,
    mths_since_last_record INT,

    open_acc INT,
    pub_rec INT,
    revol_bal DECIMAL(15,2),
    revol_util DECIMAL(10,4),
    total_acc INT,

    initial_list_status VARCHAR(10),

    out_prncp DECIMAL(15,2),
    out_prncp_inv DECIMAL(15,2),
    total_pymnt DECIMAL(15,2),
    total_pymnt_inv DECIMAL(15,2),
    total_rec_prncp DECIMAL(15,2),
    total_rec_int DECIMAL(15,2),
    total_rec_late_fee DECIMAL(15,2),
    recoveries DECIMAL(15,2),
    collection_recovery_fee DECIMAL(15,2),

    last_pymnt_d VARCHAR(20),
    last_pymnt_amnt DECIMAL(15,2),
    next_pymnt_d VARCHAR(20),
    last_credit_pull_d VARCHAR(20),

    collections_12_mths_ex_med INT,
    mths_since_last_major_derog INT,

    policy_code INT,
    application_type VARCHAR(50),

    joint_annual_income DECIMAL(15,2),
    dti_joint DECIMAL(10,4),
    verification_status_joint VARCHAR(50),

    acc_now_delinq INT,
    tot_coll_amt DECIMAL(15,2),
    tot_cur_bal DECIMAL(15,2),

    open_acc_6m INT,
    open_act_il INT,
    open_il_12m INT,
    open_il_24m INT,
    mths_since_rcnt_il INT,
    total_bal_il DECIMAL(15,2),
    il_util DECIMAL(10,4),

    open_rv_12m INT,
    open_rv_24m INT,
    max_bal_bc DECIMAL(15,2),
    all_util DECIMAL(10,4),
    total_rev_hi_lim DECIMAL(15,2),

    inq_fi INT,
    total_cu_tl INT,
    inq_last_12m INT,
    acc_open_past_24mths INT,

    avg_cur_bal DECIMAL(15,2),
    bc_open_to_buy DECIMAL(15,2),
    bc_util DECIMAL(10,4),

    chargeoff_within_12_mths INT,
    delinq_amnt DECIMAL(15,2),

    mo_sin_old_il_acct INT,
    mo_sin_old_rev_tl_op INT,
    mo_sin_rcnt_rev_tl_op INT,
    mo_sin_rcnt_tl INT,

    mort_acc INT,

    mths_since_recent_bc INT,
    mths_since_recent_bc_dlq INT,
    mths_since_recent_inq INT,
    mths_since_recent_revol_delinq INT,

    num_accts_ever_120_pd INT,
    num_actv_bc_tl INT,
    num_actv_rev_tl INT,
    num_bc_sats INT,
    num_bc_tl INT,
    num_il_tl INT,
    num_op_rev_tl INT,
    num_rev_accts INT,
    num_rev_tl_bal_gt_0 INT,
    num_sats INT,

    num_tl_120dpd_2m INT,
    num_tl_30dpd INT,
    num_tl_90g_dpd_24m INT,
    num_tl_op_past_12m INT,

    pct_tl_nvr_dlq DECIMAL(10,4),
    percent_bc_gt_75 DECIMAL(10,4),

    pub_rec_bankruptcies INT,
    tax_liens INT,

    tot_hi_cred_lim DECIMAL(15,2),
    total_bal_ex_mort DECIMAL(15,2),
    total_bc_limit DECIMAL(15,2),
    total_il_high_credit_limit DECIMAL(15,2),

    revol_bal_joint DECIMAL(15,2),

    sec_app_earliest_cr_line VARCHAR(20),
    sec_app_inq_last_6mths INT,
    sec_app_mort_acc INT,
    sec_app_open_acc INT,
    sec_app_revol_util DECIMAL(10,4),
    sec_app_open_act_il INT,
    sec_app_num_rev_accts INT,
    sec_app_chargeoff_within_12_mths INT,
    sec_app_collections_12_mths_ex_med INT,
    sec_app_mths_since_last_major_derog INT,

    hardship_flag VARCHAR(10),
    hardship_type VARCHAR(100),
    hardship_reason VARCHAR(255),
    hardship_status VARCHAR(100),

    deferral_term INT,
    hardship_amount DECIMAL(15,2),

    hardship_start_date VARCHAR(20),
    hardship_end_date VARCHAR(20),
    payment_plan_start_date VARCHAR(20),

    hardship_length INT,
    hardship_dpd INT,

    hardship_loan_status VARCHAR(100),

    orig_projected_additional_accrued_interest DECIMAL(15,2),
    hardship_payoff_balance_amount DECIMAL(15,2),
    hardship_last_payment_amount DECIMAL(15,2),

    disbursement_method VARCHAR(50),

    debt_settlement_flag VARCHAR(10),
    debt_settlement_flag_date VARCHAR(20),

    settlement_status VARCHAR(50),
    settlement_date VARCHAR(20),

    settlement_amount DECIMAL(15,2),
    settlement_percentage DECIMAL(10,4),
    settlement_term INT,

    CONSTRAINT fk_loan_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id),

    INDEX idx_customer_id (customer_id),
    INDEX idx_loan_status (loan_status),
    INDEX idx_grade (grade),
    INDEX idx_issue_date (issue_d)
);