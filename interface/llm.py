from openai import OpenAI
import dotenv
import os

import pandas as pd


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # if the key already exists in the environment variables, it will use that, otherwise it will use the .env file to get the key
if not OPENAI_API_KEY:
    dotenv.load_dotenv(".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


FILTER_INSTRUCTION = """
Your are an idea filter is designed to weed out ideas that are sloppy, off-topic (i.e., not sustainability related), unsuitable, or vague (such as the over-generic content that prioritizes form over substance, offering generalities instead of specific details). This filtration system helps concentrate human evaluators\' time and resources on concepts that are meticulously crafted, well-articulated, and hold tangible relevance.
The intended audience is venture capitalists.
When evaluating ideas you will first concisely evaluate it on each criterion, then output a word of recommendation: "pass" or "fail" for each criterion and overall
The output format will be like:
<evaluation category 1>: ... </PASS/ or /FAIL/>
<evaluation category 2>: ... </PASS/ or /FAIL/>
<...>
FEASIBILITY: <score out of 5>
INNOVATION: <score out of 5>
OVERALL: /PASS/ or /FAIL/"
"""

SUMMARY_INSTRUCTION = """
Your are an idea summarizer is designed to summarize the idea into a sentence.
The intended audience is venture capitalists.
Please note that the summary should be concise and to the point, and should not include any information that is not already in the idea.
"""

class LLM():
    def __init__(self):
        self.client = OpenAI()

    def get_response(self, instruction, item):
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                "role": "system",
                "content": instruction
                },
                {
                "role": "user",
                "content": item
                }
            ],
            temperature=0.7,
            max_tokens=None,
            top_p=1
            )
        
        # extract message from response
        message = response.choices[0].message.content
        return message
    
    def idea_to_prompt(self, idea):
        # idea is a df row with columns question, solution
        question = idea['problem']
        solution = idea['solution']
        item = "PROBLEM: " + str(question) + "\nSOLUTION: " + str(solution)
        return item
    
    def get_filter_response(self, idea):
        item = self.idea_to_prompt(idea)
        return self.get_response(FILTER_INSTRUCTION, item)
    
    def get_summary_response(self, idea):
        item = self.idea_to_prompt(idea)
        return self.get_response(SUMMARY_INSTRUCTION, item)
    
    def filter(self, idea):
        """
        returns a dictionary
        """
        response = self.get_filter_response(idea)
        # extract the last /PASS/ or /FAIL/ from the response
        last_pass = response.rfind("/PASS/")
        last_fail = response.rfind("/FAIL/")
        last = max(last_pass, last_fail)
        # if no pass or fail, return true
        if last == -1:
            pass_fail = "/PASS/"
        else:
            # extract the pass/fail
            pass_fail = response[last:last+6]
        # extract innovation and feasibility scores
        innovation_score = extract_score(response, "INNOVATION: ")
        feasibility_score = extract_score(response, "FEASIBILITY: ")
        return {
            "passed": pass_fail == "/PASS/",
            "response": response,
            "innovation_score": innovation_score,
            "feasibility_score": feasibility_score,
        }


def extract_score(response, keyword):
            """
            extracts a score from the response
            the score is the number after the keyword: keyword <score>
            if no score is found, return None
            """
            keyword_index = response.find(keyword)
            if keyword_index == -1:
                return None
            score_index = keyword_index + len(keyword)
            score = response[score_index:score_index+1]
            # try to convert to int, if not possible, return None
            try:
                score = int(score)
            except ValueError:
                return None
            return score