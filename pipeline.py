from agents import (
    build_fetch_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    router_chain,
    llm
)


# ==================================================
# RESEARCH PIPELINE
# ==================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # --------------------------------------------------
    # STEP 1 - SEARCH AGENT
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("STEP 1 - SEARCH AGENT IS WORKING")
    print("=" * 50)

    try:
        search_agent = build_search_agent()

        search_result = search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
Find recent, reliable, and detailed information about:

{topic}

Search the web and provide the most relevant sources,
including their titles, URLs, and useful information.
"""
                )
            ]
        })

        state["search_results"] = search_result["messages"][-1].content

    except Exception as e:
        state["search_results"] = f"Search failed: {str(e)}"

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"])


    # --------------------------------------------------
    # STEP 2 - READER / FETCH AGENT
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("STEP 2 - READER AGENT IS SCRAPING SOURCES")
    print("=" * 50)

    try:
        reader_agent = build_fetch_agent()

        reader_result = reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"""
Based on the following search results about:

{topic}

Identify the most relevant URL and scrape it using
the scrape_url tool.

Search Results:
{state["search_results"][:3000]}

Return the useful factual information obtained from
the source.
"""
                )
            ]
        })

        state["scraped_content"] = (
            reader_result["messages"][-1].content
        )

    except Exception as e:
        state["scraped_content"] = (
            f"Source reading failed: {str(e)}"
        )

    print("\nSCRAPED CONTENT:\n")
    print(state["scraped_content"])


    # --------------------------------------------------
    # STEP 3 - WRITER
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("STEP 3 - WRITER IS DRAFTING THE REPORT")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )

    try:
        writer_report = writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })

        state["report"] = writer_report

    except Exception as e:
        state["report"] = (
            f"Report generation failed: {str(e)}"
        )

    print("\nFINAL REPORT:\n")
    print(state["report"])


    # --------------------------------------------------
    # STEP 4 - CRITIC
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("STEP 4 - CRITIC IS REVIEWING THE REPORT")
    print("=" * 50)

    try:
        critic_result = critic_chain.invoke({
            "report": state["report"]
        })

        state["critic_report"] = critic_result

    except Exception as e:
        state["critic_report"] = (
            f"Critic review failed: {str(e)}"
        )

    print("\nCRITIC REPORT:\n")
    print(state["critic_report"])

    return state


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():

    print("\n" + "=" * 50)
    print("MULTI-AGENT AI RESEARCH ASSISTANT")
    print("=" * 50)

    print("\nType 'exit' or 'quit' to stop.")

    while True:

        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter something.")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break


        # --------------------------------------------------
        # ROUTER
        # --------------------------------------------------

        try:

            route_decision = router_chain.invoke({
                "query": user_input
            })

            route = route_decision.category

        except Exception as e:

            print(f"\nRouter error: {e}")
            continue


        print(f"\nRoute: {route}")


        # --------------------------------------------------
        # RESEARCH
        # --------------------------------------------------

        if route == "RESEARCH":

            result = run_research_pipeline(user_input)

            print("\n" + "=" * 50)
            print("FINAL REPORT")
            print("=" * 50)

            print(result["report"])

            print("\n" + "=" * 50)
            print("CRITIC REVIEW")
            print("=" * 50)

            print(result["critic_report"])


        # --------------------------------------------------
        # CASUAL
        # --------------------------------------------------

        elif route == "CASUAL":

            try:

                response = llm.invoke(user_input)

                print("\nAssistant:", response.content)

            except Exception as e:

                print(f"\nAssistant error: {e}")


        # --------------------------------------------------
        # UNKNOWN ROUTE
        # --------------------------------------------------

        else:

            print(
                "\nUnable to determine the request type."
            )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()

