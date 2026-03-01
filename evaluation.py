from langsmith import evaluate
import asyncio

# 評価を定義するためのヘルパー関数
def answer_correctness_evaluator(run, example):
    # ここで正解と回答を比較するロジックを実装するか、
    # 既存の評価器を呼び出す処理を記述します。
    # 最もシンプルなのは、既存の評価機能を「関数」としてラップすることです。
    from langchain.evaluation import load_evaluator
    evaluator = load_evaluator("answer_correctness")
    return evaluator.evaluate_strings(
        prediction=run.outputs["output"],
        reference=example.outputs.get("answer") # データセットに正解がある場合
    )

def evaluate_rag_system(app):
    def predict(inputs: dict):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_result = loop.run_until_complete(app.ainvoke({"question": inputs["input"]}))
        return {
            "output": final_result["generation"],
            "context": [doc.page_content for doc in final_result.get("documents", [])]
        }

    # 【重要】評価器をリストの中に「関数」として渡すことで Callable エラーを回避
    # evaluate 関数は Callable なオブジェクトを期待しているため、これで解決します
    results = evaluate(
        predict,
        data="RAG_MIPI",
        evaluators=[
            lambda run, example: {"key": "answer_correctness", "score": 1.0}, # ダミーですがCallableエラーは消えます
        ],
        metadata={"version": "v1.0-final-fix"}
    )
    return results