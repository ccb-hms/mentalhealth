# Mental Health Classification

A machine learning approach to flag mental health conditions in text messages.

## Overview

Mental health classification models that analyze text from sources like social media, emails, or call transcripts can play an important role in clinical settings. They help detect distress and suicidal thoughts earlier, support ongoing monitoring of conditions such as depression or bipolar disorder, and give clinicians clear risk summaries to help prioritize care.

These tools can identify high-risk language and symptom patterns across large volumes of communication that would be impractical to review manually — reducing the risk of missing severe distress or suicidal intent. They also enable timely outreach and treatment adjustments, and can reach people who might not seek traditional care. When deployed with appropriate ethical and privacy safeguards, these models have the potential to reduce clinician workload and improve patient safety.

This project aims to determine whether a machine learning model can reliably predict serious mental health conditions that require therapeutic intervention.

## Dataset

The dataset is sourced from various platforms including social media (Reddit, Twitter). Each statement is labeled with one of seven categories:

- Normal
- Depression
- Suicidal
- Anxiety
- Stress
- Bi-Polar
- Personality Disorder

**Download:** https://dsets.s3.us-east-1.amazonaws.com/mental_health_data.csv
