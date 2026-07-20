import time

import anthropic
from tqdm.auto import tqdm
from rag_helper import RAGBase


# 429 (rate limit), 5xx (server error) and 529 (overloaded) are transient and
# safe to retry; 4xx like 400/404 are our fault and should surface immediately.
def _is_transient(exc):
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-5":   {"input": 3.00, "output": 15.00}, 
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00},
    "claude-opus-4-8":   {"input": 5.00, "output": 25.00},
}


def calc_price(usage, model="claude-haiku-4-5"):
    price = PRICING[model]

    input_cost = (usage.input_tokens / 1_000_000) * price["input"]
    output_cost = (usage.output_tokens / 1_000_000) * price["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages, model="claude-haiku-4-5"):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage, model=model)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model="claude-haiku-4-5"):
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=instructions,
        messages=messages,
        output_format=output_type
    )

    return response.parsed_output, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="claude-haiku-4-5",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt, max_retries=5):
        input_messages = [
            {"role": "user", "content": prompt}
        ]

        for attempt in range(max_retries):
            try:
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.instructions,
                    messages=input_messages
                )
                break
            except Exception as exc:
                if attempt == max_retries - 1 or not _is_transient(exc):
                    raise
                time.sleep(2 ** attempt)

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return next(b.text for b in response.content if b.type == "text")

    def total_cost(self):
        return calc_total_price(self.usages, model=self.model)


def map_progress_safe(pool, seq, f):
    """Like map_progress, but never loses completed work.

    Returns (results, errors) where results holds every item that succeeded and
    errors holds (element, exception) pairs for the ones that failed. A single
    out-of-credits/rate-limit failure no longer discards the whole run.
    """
    results = []
    errors = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append((el, future))

        for el, future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                errors.append((el, e))

    return results, errors
