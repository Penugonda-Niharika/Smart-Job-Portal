"""
Automated Comprehensive Test Suite for Smart Job Portal.
Executes real input/output verification across authentication, role permissions,
job CRUD, filtering, AI/ML features, and responsive data integrity.
"""

import portal_test_env  # noqa: F401  (must come first: points the app at a temporary copy of data/)
import unittest
import json
import io
import app as app_module
from app import app
from services.recommendations import get_recommendations, compute_skill_match
from services.resume_analyzer import (
    extract_text_from_file, analyze_resume_against_job, extract_skills_from_text
)
from services.career_guidance import analyze_career_gap
from services.job_risk_checker import evaluate_job_risk
from werkzeug.datastructures import FileStorage
from werkzeug.security import check_password_hash


class SmartJobPortalComprehensiveTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False  # Disable CSRF during test calls
        self.client = app.test_client()

    # --------------------------------------------------------------------------
    # 1. Authentication & Security
    # --------------------------------------------------------------------------
    def test_01_user_signup_seeker(self):
        """Test candidate user registration with password hashing."""
        response = self.client.post("/signup", data={
            "name": "Jane Candidate",
            "email": "jane.candidate@test.com",
            "role": "seeker",
            "password": "securepassword123",
            "confirm_password": "securepassword123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        user = next((u for u in app_module.RUNTIME_USERS if u["email"] == "jane.candidate@test.com"), None)
        self.assertIsNotNone(user)
        self.assertTrue(check_password_hash(user["password_hash"], "securepassword123"))
        self.assertEqual(user["role"], "seeker")

    def test_02_user_signup_employer(self):
        """Test employer registration with company details."""
        response = self.client.post("/signup", data={
            "name": "Robert Recruiter",
            "email": "robert@innovatetech.test",
            "role": "employer",
            "company_name": "InnovateTech Corp",
            "password": "employerpass123",
            "confirm_password": "employerpass123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        user = next((u for u in app_module.RUNTIME_USERS if u["email"] == "robert@innovatetech.test"), None)
        self.assertIsNotNone(user)
        self.assertEqual(user["company_name"], "InnovateTech Corp")
        self.assertEqual(user["role"], "employer")

    def test_03_login_valid_and_invalid(self):
        """Test authentication for valid and invalid credentials."""
        # Invalid password
        bad_resp = self.client.post("/login", data={
            "email": "seeker@example.com",
            "password": "wrongpassword"
        }, follow_redirects=True)
        self.assertIn(b"Invalid email or password", bad_resp.data)

        # Valid password
        good_resp = self.client.post("/login", data={
            "email": "seeker@example.com",
            "password": "seeker123"
        }, follow_redirects=True)
        self.assertEqual(good_resp.status_code, 200)
        self.assertIn(b"Candidate Dashboard", good_resp.data)

    def test_04_role_access_control(self):
        """Verify that job seekers cannot access employer or admin dashboards."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-101"
            sess["user_role"] = "seeker"
            sess["user_email"] = "seeker@example.com"

        # Seeker accessing employer dashboard should be forbidden (403)
        emp_resp = self.client.get("/dashboard/employer")
        self.assertEqual(emp_resp.status_code, 403)

        # Seeker accessing admin dashboard should be forbidden (403)
        admin_resp = self.client.get("/dashboard/admin")
        self.assertEqual(admin_resp.status_code, 403)

    # --------------------------------------------------------------------------
    # 2. Strict Employment Types & Job Exploration
    # --------------------------------------------------------------------------
    def test_05_strict_employment_types_data(self):
        """Verify that every single job in the catalog is strictly Full-time or Part-time."""
        allowed = {"Full-time", "Part-time"}
        for j in app_module.RUNTIME_JOBS:
            self.assertIn(
                j.get("employment_type"), allowed,
                f"Job {j.get('id')} has invalid type: {j.get('employment_type')}"
            )

    def test_06_jobs_filtering(self):
        """Test search and independent filtering by type and arrangement."""
        # Filter full-time
        resp_ft = self.client.get("/jobs?employment_type=Full-time")
        self.assertEqual(resp_ft.status_code, 200)
        self.assertIn(b"Full-time", resp_ft.data)

        # Filter part-time
        resp_pt = self.client.get("/jobs?employment_type=Part-time")
        self.assertEqual(resp_pt.status_code, 200)
        self.assertIn(b"Part-time", resp_pt.data)

        # Filter Remote
        resp_remote = self.client.get("/jobs?work_arrangement=Remote")
        self.assertEqual(resp_remote.status_code, 200)

    def test_07_job_detail_and_links(self):
        """Test that job details page renders full attributes and external link if present."""
        first_job = app_module.RUNTIME_JOBS[0]
        resp = self.client.get(f"/jobs/{first_job['id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(first_job["title"].encode("utf-8"), resp.data)
        self.assertIn(b"Key Responsibilities", resp.data)
        self.assertIn(b"Required Skills", resp.data)

        if first_job.get("application_url"):
            self.assertIn(b"Apply on Company / LinkedIn Site", resp.data)

    # --------------------------------------------------------------------------
    # 3. Employer CRUD Operations
    # --------------------------------------------------------------------------
    def test_08_employer_job_crud(self):
        """Test job creation, editing, and deletion by an employer."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-102"
            sess["user_role"] = "employer"
            sess["company_name"] = "Apex Cloud Systems"

        # 1. Create Job
        create_resp = self.client.post("/jobs/create", data={
            "title": "Lead Python Distributed Architect",
            "company": "Apex Cloud Systems",
            "location": "Bengaluru, India",
            "employment_type": "Full-time",
            "work_arrangement": "Hybrid",
            "experience_level": "Senior-level",
            "salary_min": 2500000,
            "salary_max": 3500000,
            "salary_display": "₹25L - ₹35L / year",
            "description": "Architect high performance microservices.",
            "required_skills": "Python, Redis, Kafka, Docker",
            "responsibilities": "Design microservices\nLead engineering",
            "application_url": "https://careers.apexcloudsystems.example.com/lead-architect",
            "contact_email": "careers@apexcloudsystems.example.com"
        }, follow_redirects=True)
        self.assertEqual(create_resp.status_code, 200)

        created_job = next((j for j in app_module.RUNTIME_JOBS if j["title"] == "Lead Python Distributed Architect"), None)
        self.assertIsNotNone(created_job)
        self.assertEqual(created_job["employment_type"], "Full-time")

        # 2. Edit Job
        edit_resp = self.client.post(f"/jobs/{created_job['id']}/edit", data={
            "title": "Principal Python Systems Architect",
            "company": "Apex Cloud Systems",
            "location": "Bengaluru, India",
            "employment_type": "Full-time",
            "work_arrangement": "Remote",
            "experience_level": "Senior-level",
            "salary_min": 3000000,
            "salary_max": 4000000,
            "salary_display": "₹30L - ₹40L / year",
            "description": "Updated architecture description.",
            "required_skills": "Python, Redis, Kafka, Kubernetes",
            "responsibilities": "Lead systems design",
            "application_url": "https://careers.apexcloudsystems.example.com/lead-architect",
            "contact_email": "careers@apexcloudsystems.example.com"
        }, follow_redirects=True)
        self.assertEqual(edit_resp.status_code, 200)
        self.assertEqual(created_job["title"], "Principal Python Systems Architect")

        # 3. Delete Job
        delete_resp = self.client.post(f"/jobs/{created_job['id']}/delete", follow_redirects=True)
        self.assertEqual(delete_resp.status_code, 200)
        # Read the live list: the delete route rebinds app.RUNTIME_JOBS, so an earlier reference would be stale.
        self.assertIsNone(next((j for j in app_module.RUNTIME_JOBS if j["id"] == created_job["id"]), None))
        on_disk = json.loads((portal_test_env.DATA_DIR / "jobs.json").read_text(encoding="utf-8"))
        self.assertFalse(any(j["id"] == created_job["id"] for j in on_disk))

    # --------------------------------------------------------------------------
    # 4. Job Applications & Status Updates
    # --------------------------------------------------------------------------
    def test_09_applications_flow(self):
        """Test candidate application submission, duplicate blocking, and status updates."""
        # The status-updating employer (USR-102) may only manage applications on jobs they own.
        target_job = next(j for j in app_module.RUNTIME_JOBS if j.get("employer_id") == "USR-102")

        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-TEST-APPLICANT"
            sess["user_role"] = "seeker"
            sess["user_name"] = "Test Candidate"
            sess["user_email"] = "candidate.applicant@test.com"

        # Apply
        app_resp = self.client.post(f"/jobs/{target_job['id']}/apply", data={
            "applicant_name": "Test Candidate",
            "applicant_email": "candidate.applicant@test.com",
            "cover_note": "I have solid Python and Machine Learning experience.",
            "portfolio_url": "https://github.com/testcandidate"
        }, follow_redirects=True)
        self.assertEqual(app_resp.status_code, 200)

        app_record = next((a for a in app_module.RUNTIME_APPLICATIONS if a["applicant_email"] == "candidate.applicant@test.com"), None)
        self.assertIsNotNone(app_record)
        self.assertEqual(app_record["status"], "Submitted")

        # Duplicate submission check
        dup_resp = self.client.post(f"/jobs/{target_job['id']}/apply", data={
            "applicant_name": "Test Candidate",
            "applicant_email": "candidate.applicant@test.com"
        }, follow_redirects=True)
        self.assertIn(b"already submitted an application", dup_resp.data)

        # Employer updates status to Shortlisted
        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-102"
            sess["user_role"] = "employer"
            sess["company_name"] = target_job["company"]

        status_resp = self.client.post(f"/employer/application/{app_record['id']}/status", data={
            "status": "Shortlisted"
        }, follow_redirects=True)
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(app_record["status"], "Shortlisted")

    # --------------------------------------------------------------------------
    # 5. AI / ML Features Verification
    # --------------------------------------------------------------------------
    def test_10_resume_text_extraction(self):
        """Verify text extraction on valid and invalid files."""
        # Valid TXT resume
        txt_bytes = b"Jane Doe - Software Engineer with Python, Flask, SQL, Git, and Docker experience."
        storage_txt = FileStorage(stream=io.BytesIO(txt_bytes), filename="resume.txt")
        text, err = extract_text_from_file(storage_txt)
        self.assertIsNone(err)
        self.assertIn("Python", text)

        # Unsupported file extension
        unsupported = FileStorage(stream=io.BytesIO(b"binary"), filename="resume.docx")
        text_bad, err_bad = extract_text_from_file(unsupported)
        self.assertIn("Unsupported file extension", err_bad)

        # Empty file
        empty = FileStorage(stream=io.BytesIO(b""), filename="empty.txt")
        text_empty, err_empty = extract_text_from_file(empty)
        self.assertIn("empty", err_empty)

    def test_11_tfidf_cosine_similarity(self):
        """Test scikit-learn TF-IDF cosine similarity between resume and job description."""
        resume = "Senior Python engineer building Flask web applications, Docker containers, and SQL queries."
        job_desc = "Seeking Python Developer with Flask, REST APIs, SQL, and Docker containerization experience."
        skills = ["Python", "Flask", "SQL", "Docker", "Kubernetes"]

        result = analyze_resume_against_job(resume, job_desc, skills)
        # Headline score is the skill-match percentage: 4 of the 5 required skills are present.
        self.assertEqual(result["similarity_score"], 80.0)
        self.assertEqual(result["skill_match_percentage"], 80.0)
        self.assertEqual(result["score_basis"], "skill_match")
        self.assertGreater(result["tfidf_similarity"], 0)  # text similarity still computed separately
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("Kubernetes", result["missing_skills"])
        self.assertTrue(len(result["suggestions"]) > 0)

    def test_12_skill_recommendations(self):
        """Test recommendation engine ranking logic."""
        user_skills = ["Python", "Machine Learning", "scikit-learn"]
        recs = get_recommendations(user_skills, app_module.RUNTIME_JOBS, min_score=10, limit=5)
        self.assertTrue(len(recs) > 0)
        # Highest match should have high score
        self.assertGreaterEqual(recs[0]["score"], recs[-1]["score"])

    def test_13_career_roadmap(self):
        """Test career gap analysis and structured phased learning roadmaps."""
        gap = analyze_career_gap(["Python"], "python_developer", app_module.RUNTIME_JOBS)
        self.assertEqual(gap["role_title"], "Python Developer")
        self.assertIn("Python", gap["possessed_skills"])
        self.assertTrue(len(gap["missing_skills"]) > 0)
        self.assertEqual(len(gap["roadmap"]), 4)

    def test_14_suspicious_job_checker(self):
        """Test heuristic scam detection engine on normal vs suspicious jobs."""
        normal_job = app_module.RUNTIME_JOBS[0]
        normal_eval = evaluate_job_risk(normal_job)
        self.assertEqual(normal_eval["risk_level"], "Low")

        # Test job with upfront payment and personal email
        suspicious_job = next((j for j in app_module.RUNTIME_JOBS if "Deposit Required" in j["title"]), None)
        self.assertIsNotNone(suspicious_job)
        scam_eval = evaluate_job_risk(suspicious_job)
        self.assertEqual(scam_eval["risk_level"], "High")
        self.assertTrue(len(scam_eval["warnings"]) >= 2)


if __name__ == "__main__":
    unittest.main()
