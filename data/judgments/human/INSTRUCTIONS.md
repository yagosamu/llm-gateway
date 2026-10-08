# How to grade

Open sheet.csv. For each row, read the request, the reference answer and the candidate answer, and write yes or no in the `acceptable` column. The `note` column is optional. Do not open key.json until you are done: it says which model wrote each candidate.

The judges get exactly this rubric:

You grade whether a CANDIDATE answer is an acceptable substitute for a REFERENCE answer to the same request.

Acceptable means that the person who sent the request would be about as well served by the candidate as by the reference:
- It does what the request asks, including any format, length or style constraint stated in the request.
- It makes no factual claim that contradicts the reference or the context given in the request, and invents no fact that matters to the request.
- It is complete. An answer that stops in the middle is not acceptable.

Do not reject for differences in wording, style, order or length when the request sets no constraint on them.
For open-ended requests (brainstorming, creative writing, opinions) there is no single right answer: judge whether the candidate is an on-task answer of similar usefulness, not whether it resembles the reference.
The reference can be wrong. If the candidate is right where the reference is wrong, the candidate is acceptable.

Give your reason first, in one or two sentences, then the verdict.
