from openai import OpenAI

import pandas as pd


FILTER_INSTRUCTION = """
Your are an idea filter is designed to weed out ideas that are sloppy, off-topic (i.e., not sustainability related), unsuitable, or vague (such as the over-generic content that prioritizes form over substance, offering generalities instead of specific details). This filtration system helps concentrate human evaluators\' time and resources on concepts that are meticulously crafted, well-articulated, and hold tangible relevance.
The intended audience is venture capitalists.
When evaluating ideas you will first concisely evaluate it on each criterion, then output a word of recommendation: "pass" or "fail" for each criterion and overall
The output format will be like:
<evaluation category 1>: ... /PASS/ or /FAIL/
<evaluation category 2>: ... /PASS/ or /FAIL/
<...>
OVERALL: /PASS/ or /FAIL/"
"""

SUMMARY_INSTRUCTION = """
Your are an idea summarizer is designed to summarize the idea into 1-2 sentences.
The intended audience is venture capitalists.
Please note that the summary should be concise and to the point, and should not include any information that is not already in the idea.
"""

class LLM():
    def __init__(self, api_key):
        self.client = OpenAI(api_key)

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
        message = response.choices[0].text
        return message
    
    def idea_to_prompt(self, idea):
        # idea is a df row with columns question, solution
        question = idea['question']
        solution = idea['solution']
        item = "QUESTION: " + question + "\nSOLUTION: " + solution
        return item
    
    def get_filter_response(self, idea):
        # idea is a df row with columns question, solution
        item = self.idea_to_prompt(idea)
        return self.get_response(FILTER_INSTRUCTION, item)
    
    def get_summary_response(self, idea):
        # idea is a df row with columns question, solution
        item = self.idea_to_prompt(idea)
        return self.get_response(SUMMARY_INSTRUCTION, item)
    
    def filter(self, idea):
        """
        returns a tuple: (BOOL(pass or not), message)
        """
        # idea is a df row with columns question, solution
        response = self.get_filter_response(idea)
        # extract the last /PASS/ or /FAIL/ from the response
        last_pass = response.rfind("/PASS/")
        last_fail = response.rfind("/FAIL/")
        last = max(last_pass, last_fail)
        # if no pass or fail, return true
        if last == -1:
            return (True, response)
        # extract the pass/fail
        pass_fail = response[last:last+6]
        return (pass_fail == "/PASS/", response)
