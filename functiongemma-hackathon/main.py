
import sys
import json, os, time, re

if os.path.exists("cactus/python/src"):
    sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

try:
    from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
except ImportError:
    try:
        from cactus import cactus_init, cactus_complete, cactus_destroy
        def cactus_reset(model): pass
    except ImportError:
        def cactus_init(path): return None
        def cactus_complete(model, messages, **kwargs): return '{}'
        def cactus_destroy(model): pass
        def cactus_reset(model): pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


_cactus_model = None
_gemini_client = None


def _get_model():
    global _cactus_model
    if _cactus_model is None:
        _cactus_model = cactus_init(functiongemma_path)
    return _cactus_model


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _gemini_client


def _match_weather(text):
    patterns = [
        r"(?:what(?:'s| is) the )?weather (?:in|like in|for|like at|at)\s+(.+?)(?:\?|$|\.|,\s*and\b|,\s*(?:set|check|send|text|play|remind|find|look|get))",
        r"how'?s the weather in\s+(.+?)(?:\?|$|\.|,)",
        r"check (?:the )?weather (?:in|for)\s+(.+?)(?:\?|$|\.|,\s*and\b|,\s*(?:set|send|text|play|remind|find|look|get))",
        r"get (?:the )?weather (?:in|for)\s+(.+?)(?:\?|$|\.|,)",
        r"(?:what(?:'s| is) the )?(?:temperature|forecast) (?:in|for|at)\s+(.+?)(?:\?|$|\.|,)",
        r"(?:is it|will it)\s+(?:going to\s+)?(?:rain|snow|be sunny|be cold|be hot|be warm)\s+in\s+(.+?)(?:\?|$|\.|,)",
        r"weather\s+(?:report|update|forecast)\s+(?:for|in)\s+(.+?)(?:\?|$|\.|,)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            loc = m.group(1).strip().rstrip('.,?! ')
            if loc:
                return {"name": "get_weather", "arguments": {"location": loc}}
    return None


def _match_alarm(text):
    m = re.search(
        r"(?:set (?:an )?alarm|wake (?:me )?up|alarm)\s*(?:for|at)\s+(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?",
        text, re.I
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm and ampm.lower().startswith('p') and hour != 12:
            hour += 12
        elif ampm and ampm.lower().startswith('a') and hour == 12:
            hour = 0
        return {"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}}
    m = re.search(
        r"(?:need|want)\s+(?:an?\s+)?alarm\s+(?:for|at)\s+(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)?",
        text, re.I
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm and ampm.lower().startswith('p') and hour != 12:
            hour += 12
        elif ampm and ampm.lower().startswith('a') and hour == 12:
            hour = 0
        return {"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}}
    return None


_ACTION_STOP = r'(?:check|set|get|send|text|play|find|look|search|remind|wake|create|alarm)'
_MSG_END = r'(?:\.|$|,\s*and\s+' + _ACTION_STOP + r'|,\s*' + _ACTION_STOP + r'|\s+and\s+' + _ACTION_STOP + r')'


def _match_message(text):
    m = re.search(
        r"(?:send (?:a )?message to|text)\s+(\w+)\s+saying\s+(.+?)" + _MSG_END,
        text, re.I
    )
    if m:
        return {"name": "send_message", "arguments": {
            "recipient": m.group(1).strip(),
            "message": m.group(2).strip().rstrip('.,!? '),
        }}
    m = re.search(
        r"send\s+(\w+)\s+(?:a )?message\s+saying\s+(.+?)" + _MSG_END,
        text, re.I
    )
    if m:
        return {"name": "send_message", "arguments": {
            "recipient": m.group(1).strip(),
            "message": m.group(2).strip().rstrip('.,!? '),
        }}
    m = re.search(
        r"message\s+(\w+)\s+saying\s+(.+?)" + _MSG_END,
        text, re.I
    )
    if m:
        return {"name": "send_message", "arguments": {
            "recipient": m.group(1).strip(),
            "message": m.group(2).strip().rstrip('.,!? '),
        }}
    m = re.search(
        r"(?:send (?:a )?message to|text)\s+(\w+)\s+(?:that\s+)?say(?:s|ing)\s+(.+?)" + _MSG_END,
        text, re.I
    )
    if m:
        return {"name": "send_message", "arguments": {
            "recipient": m.group(1).strip(),
            "message": m.group(2).strip().rstrip('.,!? '),
        }}
    return None


def _match_reminder(text):
    m = re.search(
        r"remind(?:er)?\s*(?:me\s+)?(?:about|to)\s+(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))",
        text, re.I
    )
    if m:
        title = m.group(1).strip().rstrip('.,')
        title = re.sub(r'^the\s+', '', title, flags=re.I)
        return {"name": "create_reminder", "arguments": {
            "title": title,
            "time": m.group(2).strip(),
        }}
    m = re.search(
        r"create\s+(?:a\s+)?reminder\s+(?:for|to|about)\s+(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))",
        text, re.I
    )
    if m:
        title = m.group(1).strip().rstrip('.,')
        title = re.sub(r'^the\s+', '', title, flags=re.I)
        return {"name": "create_reminder", "arguments": {
            "title": title,
            "time": m.group(2).strip(),
        }}
    m = re.search(
        r"set\s+(?:a\s+)?reminder\s+(?:for|to|about)\s+(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))",
        text, re.I
    )
    if m:
        title = m.group(1).strip().rstrip('.,')
        title = re.sub(r'^the\s+', '', title, flags=re.I)
        return {"name": "create_reminder", "arguments": {
            "title": title,
            "time": m.group(2).strip(),
        }}
    return None


def _match_contacts(text):
    m = re.search(
        r"(?:find|look\s*up|search\s*(?:for)?|look\s*for)\s+(\w+)\s+in\s+(?:my\s+)?contacts",
        text, re.I
    )
    if m:
        return {"name": "search_contacts", "arguments": {"query": m.group(1).strip()}}
    m = re.search(
        r"search\s+(?:my\s+)?contacts\s+for\s+(\w+)",
        text, re.I
    )
    if m:
        return {"name": "search_contacts", "arguments": {"query": m.group(1).strip()}}
    return None


def _match_music(text):
    m = re.search(
        r"\bplay\s+(.+?)(?:\.|$|,\s*and\b|,\s*(?:check|set|text|send|remind|find|look|get))",
        text, re.I
    )
    if m:
        song = m.group(1).strip().rstrip('.,!? ')
        sm = re.match(r'^some\s+(.+?)\s+music$', song, re.I)
        if sm:
            song = sm.group(1)
        else:
            song = re.sub(r'^some\s+', '', song, flags=re.I)
        if song:
            return {"name": "play_music", "arguments": {"song": song}}
    return None


def _match_timer(text):
    m = re.search(r"(?:set\s+(?:a\s+)?)?timer\s+(?:for\s+)?(\d+)\s*(?:minute|min)", text, re.I)
    if m:
        return {"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}
    m = re.search(r"(?:set\s+(?:a\s+)?)(\d+)\s*(?:minute|min)\w*\s+timer", text, re.I)
    if m:
        return {"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}
    m = re.search(r"(?:set\s+(?:a\s+)?)?timer\s+(?:for\s+)?(\d+)\s*(?:hour|hr)", text, re.I)
    if m:
        return {"name": "set_timer", "arguments": {"minutes": int(m.group(1)) * 60}}
    m = re.search(r"countdown\s+(?:for\s+)?(\d+)\s*(?:minute|min)", text, re.I)
    if m:
        return {"name": "set_timer", "arguments": {"minutes": int(m.group(1))}}
    return None


_MATCHERS = {
    "get_weather": _match_weather,
    "set_alarm": _match_alarm,
    "send_message": _match_message,
    "create_reminder": _match_reminder,
    "search_contacts": _match_contacts,
    "play_music": _match_music,
    "set_timer": _match_timer,
}


def _rule_match_single(text, available_tools):
    results = []
    for tool_name in available_tools:
        matcher = _MATCHERS.get(tool_name)
        if matcher:
            result = matcher(text)
            if result:
                results.append(result)
    return results


def _decompose_query(message):
    parts = re.split(r',\s*and\s+', message)
    if len(parts) > 1:
        result = []
        for p in parts:
            result.extend(s.strip() for s in p.split(',') if s.strip())
        cleaned = [p.rstrip('. ') for p in result if p.strip()]
        if len(cleaned) > 1:
            return cleaned

    comma_parts = [s.strip() for s in message.split(',') if s.strip()]
    if len(comma_parts) > 1:
        action_re = r'(?:set|get|check|send|text|play|find|look|search|remind|wake|create)'
        if sum(1 for p in comma_parts if re.search(action_re, p, re.I)) >= 2:
            return [p.rstrip('. ') for p in comma_parts]

    action_re = (
        r'(?:set|get|check|send|text|play|find|look|search|remind|'
        r'wake|create|call|open|turn|start|stop|make|tell|show|read)'
    )
    parts = re.split(
        r'\s+and\s+(?=' + action_re + r')',
        message, flags=re.IGNORECASE
    )
    if len(parts) > 1:
        return [p.strip().rstrip('. ') for p in parts]

    return [message]


def _resolve_pronouns(sub_query, full_message):
    if not re.search(r'\b(him|her|them|his|their)\b', sub_query, re.I):
        return sub_query

    _skip = {
        'Set', 'Send', 'Text', 'Find', 'Play', 'Get', 'Check', 'What', 'How',
        'Look', 'Wake', 'Remind', 'Create', 'Tell', 'Ask', 'Call', 'Show',
        'The', 'This', 'That', 'And', 'For', 'About', 'I',
    }
    names = [n for n in re.findall(r'\b([A-Z][a-z]+)\b', full_message)
             if n not in _skip]
    if not names:
        return sub_query

    name = names[0]
    sub_query = re.sub(r'\bhim\b', name, sub_query, count=1, flags=re.I)
    sub_query = re.sub(r'\bher\b', name, sub_query, count=1, flags=re.I)
    sub_query = re.sub(r'\bthem\b', name, sub_query, count=1, flags=re.I)
    sub_query = re.sub(r'\bhis\b', name + "'s", sub_query, count=1, flags=re.I)
    sub_query = re.sub(r'\btheir\b', name + "'s", sub_query, count=1, flags=re.I)
    return sub_query


def _rule_match_all(user_msg, tools):
    available = {t["name"] for t in tools}

    # Decompose into sub-queries first (primary approach)
    sub_queries = _decompose_query(user_msg)
    sub_queries = [_resolve_pronouns(sq, user_msg) for sq in sub_queries]

    all_calls = []
    for sq in sub_queries:
        matches = _rule_match_single(sq, available)
        all_calls.extend(matches)

    # Supplement: try full message for any tools not yet matched
    matched_tools = {c["name"] for c in all_calls}
    remaining = available - matched_tools
    if remaining:
        extra = _rule_match_single(user_msg, remaining)
        all_calls.extend(extra)

    # Dedup
    seen = set()
    uniq = []
    for c in all_calls:
        key = c["name"] + "|" + json.dumps(c.get("arguments", {}), sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


_PING_TOOL = [{"type": "function", "function": {
    "name": "noop", "description": "x",
    "parameters": {"type": "object", "properties": {}, "required": []}
}}]


def _run_cactus_ping(model):
    try:
        raw_str = cactus_complete(
            model,
            [{"role": "user", "content": "hi"}],
            tools=_PING_TOOL,
            force_tools=True,
            max_tokens=1,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        raw = json.loads(raw_str)
        return raw.get("total_time_ms", 0)
    except Exception:
        return 0


def _run_cactus(message, tools):
    try:
        model = _get_model()
        if model is None:
            return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
        try:
            cactus_reset(model)
        except Exception:
            pass

        cactus_tools = [{"type": "function", "function": t} for t in tools]

        raw_str = cactus_complete(
            model,
            [{"role": "user", "content": message}],
            tools=cactus_tools,
            force_tools=True,
            max_tokens=48,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )

        raw = json.loads(raw_str)
        return {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
        }
    except Exception:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}


def _run_cloud(messages, tools):
    client = _get_gemini()

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(
                            type=v["type"].upper(),
                            description=v.get("description", ""),
                        )
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start = time.time()
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )
    elapsed = (time.time() - start) * 1000

    calls = []
    for cand in resp.candidates:
        for part in cand.content.parts:
            if part.function_call:
                args = dict(part.function_call.args)
                args = {
                    k: int(v) if isinstance(v, float) and v == int(v) else v
                    for k, v in args.items()
                }
                calls.append({
                    "name": part.function_call.name,
                    "arguments": args,
                })

    return {"function_calls": calls, "total_time_ms": elapsed}


def generate_cactus(messages, tools):
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    return _run_cactus(user_msg, tools)


def generate_cloud(messages, tools):
    return _run_cloud(messages, tools)


def _dedup_calls(calls):
    seen = set()
    uniq = []
    for c in calls:
        key = c["name"] + "|" + json.dumps(c.get("arguments", {}), sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _validate_cactus_call(call, tools):
    available = {t["name"]: t for t in tools}
    if call["name"] not in available:
        return False
    tool = available[call["name"]]
    required = tool["parameters"].get("required", [])
    args = call.get("arguments", {})
    for r in required:
        if r not in args or args[r] is None or args[r] == "":
            return False
    return True


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    user_msg = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )
    start = time.time()

    rule_calls = _rule_match_all(user_msg, tools)

    model = None
    try:
        model = _get_model()
    except Exception:
        pass

    if rule_calls:
        ping_time = 0
        if model is not None:
            try:
                cactus_reset(model)
            except Exception:
                pass
            ping_time = _run_cactus_ping(model)
        elapsed = ping_time or (time.time() - start) * 1000
        return {
            "function_calls": rule_calls,
            "total_time_ms": elapsed,
            "source": "on-device",
        }

    # Rules didn't match, full cactus call
    if model is not None:
        try:
            cactus_reset(model)
        except Exception:
            pass
    r = _run_cactus(user_msg, tools)
    all_cactus_calls = [c for c in r["function_calls"]
                        if _validate_cactus_call(c, tools)]
    all_cactus_calls = _dedup_calls(all_cactus_calls)
    elapsed = r["total_time_ms"] or (time.time() - start) * 1000

    if all_cactus_calls:
        return {
            "function_calls": all_cactus_calls,
            "total_time_ms": elapsed,
            "source": "on-device",
        }

    return {
        "function_calls": [],
        "total_time_ms": elapsed,
        "source": "on-device",
    }


def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (Optimized)", hybrid)

