"""Prompts shared by the teacher pass, baselines, and fine-tuned contestants.

The extraction instructions are IDENTICAL for every model in the bake-off —
teacher, baselines, and contestants all see the same task definition, so the
only variable is the model. Frozen alongside schema.json (2026-07-17).
"""

EXTRACTION_SYSTEM = """\
You extract compliance matrices from US state/local government RFPs.

A compliance matrix is the complete list of things a vendor must do, provide, \
or satisfy to submit a responsive proposal. Missing a single "shall" can \
disqualify a bid, so recall matters more than brevity.

Extract every requirement that meets ANY of these tests:
- Binding language: shall, must, will (vendor obligation), is required to, may not.
- Advisory language directed at the vendor: should, is encouraged to.
- Administrative facts a bidder must act on: submission deadlines, question \
deadlines, conference dates, required forms, page limits, copy counts, \
delivery addresses, eligibility/certification prerequisites, evaluation \
criteria and weights.

Rules:
1. verbatim_text is the EXACT sentence(s) from the RFP. Never paraphrase, \
never summarize, never fix typos. Trim only surrounding boilerplate.
2. One requirement per row. If a sentence contains two independent \
obligations, split it into two rows repeating the shared text.
3. section_ref is the RFP's own numbering (e.g. "3.2.1", "Section IV.B"). \
Use the nearest numbered heading if the sentence itself is unnumbered.
4. type: "shall" for binding (shall/must/will/required), "should" for \
advisory, "info" for administrative facts with no modal verb.
5. category: eligibility | deliverable | deadline | form | evaluation. \
Pick the single best fit; deadlines with dates always get category \
"deadline" and an ISO due_date.
6. page is the PDF page number where the text appears (1-indexed).
7. id values are sequential: R-001, R-002, ... in document order.
8. Do NOT invent requirements. If the document is not an RFP or has no \
requirements, return an empty requirements array.
"""

EXTRACTION_USER = """\
Solicitation ID: {rfp_id}

Extract the full compliance matrix from the attached RFP. Work through the \
document section by section; do not stop until you have covered the final \
page, including attachments and exhibits that contain vendor obligations.\
"""

# One worked example shared by every baseline and contestant (synthetic, so no
# training or gold RFP leaks into prompts). Text baselines get it as a
# user/assistant exchange before the real RFP.
FEW_SHOT_USER = """\
Solicitation ID: EXAMPLE-001

Extract the full compliance matrix from the attached RFP. Work through the \
document section by section; do not stop until you have covered the final \
page, including attachments and exhibits that contain vendor obligations.

RFP TEXT:
[page 3]
2.1 SUBMISSION
Respondents shall submit one (1) original and three (3) copies of the \
Proposal no later than 2:00 PM CT on March 5, 2026. Late submissions will \
not be accepted.
2.2 VENDOR QUALIFICATIONS
The Respondent must hold a current Texas HVAC contractor license. \
Respondents should have five (5) years of experience servicing commercial \
chillers.
[page 4]
2.3 FORMS
The Respondent shall complete and return Attachment A, Bid Certification.\
"""

FEW_SHOT_ASSISTANT = """\
{"rfp_id": "EXAMPLE-001", "requirements": [
{"id": "R-001", "verbatim_text": "Respondents shall submit one (1) original \
and three (3) copies of the Proposal no later than 2:00 PM CT on March 5, \
2026.", "section_ref": "2.1", "type": "shall", "category": "deadline", \
"due_date": "2026-03-05", "page": 3},
{"id": "R-002", "verbatim_text": "Late submissions will not be accepted.", \
"section_ref": "2.1", "type": "info", "category": "deadline", \
"due_date": null, "page": 3},
{"id": "R-003", "verbatim_text": "The Respondent must hold a current Texas \
HVAC contractor license.", "section_ref": "2.2", "type": "shall", \
"category": "eligibility", "due_date": null, "page": 3},
{"id": "R-004", "verbatim_text": "Respondents should have five (5) years of \
experience servicing commercial chillers.", "section_ref": "2.2", \
"type": "should", "category": "eligibility", "due_date": null, "page": 3},
{"id": "R-005", "verbatim_text": "The Respondent shall complete and return \
Attachment A, Bid Certification.", "section_ref": "2.3", "type": "shall", \
"category": "form", "due_date": null, "page": 4}
]}\
"""
