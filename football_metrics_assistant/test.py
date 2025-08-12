import requests
import json
import time
import random
import json
from datetime import datetime

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run football metrics API tests")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON output")
    parser.add_argument("--json-file", help="Custom JSON output filename")
    return parser.parse_args()

# API endpoint
API_URL = "http://localhost:8000/chat"

# Comprehensive test cases covering all functionality
test_cases = {
    "BASIC_TOP_N": [
        "top 5 players by goals per 90",
        "best 10 players by assists",
        "highest xG per 90",
        "most goals",
        "top 3 strikers by npxG per 90",
        "best 7 players by xA per 90",
        "top players by saves per 90",
        "highest clean sheets"
    ],
    
    "LEAGUE_SPECIFIC": [
        "most goals in Premier League",
        "top 5 assists in La Liga", 
        "highest xG per 90 in Serie A",
        "best players in Bundesliga by npxG",
        "top 3 strikers in Ligue 1",
        "most saves in Eredivisie",
        "highest xA in MLS",
        "best defenders in Championship"
    ],
    
    "POSITION_FILTERING": [
        "top 5 strikers by goals per 90",
        "best midfielders by assists per 90",
        "highest xG for wingers",
        "best goalkeepers by saves per 90",
        "top defenders by tackles per 90",
        "most goals for centre-backs",
        "best full-backs by crosses per 90",
        "top forwards by npxG per 90"
    ],
    
    "AGE_FILTERS": [
        "top 5 players under 23 by goals per 90",
        "best U21 players by xG per 90",
        "highest assists for players over 30",
        "U25 strikers with most goals",
        "players under 20 with highest xA",
        "best U23 midfielders by npxG",
        "top young players by goals + assists",
        "highest scoring players over 35"
    ],
    
    "MINUTES_FILTERING": [
        "players with 500+ minutes",
        "top 5 players by goals with more than 1000 minutes",
        "how many players with at least 2000 minutes",
        "strikers in Premier League with 500+ minutes",
        "players with over 1500 minutes by assists",
        "best players with 2000+ minutes by xG",
        "under 23 players with 1000+ minutes",
        "goalkeepers with at least 900 minutes"
    ],
    
    "STAT_VALUE_FILTERS": [
        "players with more than 0.5 goals per 90",
        "players with at least 3 assists per 90",
        "strikers in La Liga with more than 0.3 goals per 90",
        "midfielders under 25 with at least 2 assists per 90",
        "players with over 1.0 xG per 90",
        "wingers with more than 0.8 xA per 90",
        "defenders with at least 5 tackles per 90",
        "goalkeepers with over 70% save percentage"
    ],
    
    "FORMULA_QUERIES": [
        "top 5 players by xG + xA",
        "best players by goals + assists per 90",
        "highest xG + xA per 90 in Premier League",
        "top 10 players by Goals + Assists per 90 in top 5 leagues",
        "best strikers by goals / xG ratio",
        "highest goals / xg in premier league",
        "top players by npxG + xA per 90",
        "best midfielders by assists + key passes",
        "highest xG / shots ratio",
        "top defenders by tackles + interceptions",
        "best players by goals - penalties",
        "strikers by (goals + assists) / minutes * 90"
    ],
    
    "MULTI_LEAGUE": [
        "top 5 players by goals per 90 in top 5 leagues",
        "best strikers in Premier League and La Liga",
        "highest xG per 90 in top 5 leagues for players under 23",
        "top 10 players in Premier League and Serie A by assists",
        "best midfielders in Bundesliga and Ligue 1",
        "most goals in Premier League and La Liga",
        "top defenders in top 5 leagues by tackles",
        "highest xA in Serie A and Bundesliga"
    ],
    
    "COUNT_QUERIES": [
        "how many players in Premier League",
        "number of players in Serie A",
        "how many strikers",
        "how many players under 23 in La Liga",
        "number of strikers with more than 0.5 goals per 90",
        "how many players in Bundesliga with 500+ minutes",
        "count of midfielders over 25",
        "how many goalkeepers in top 5 leagues",
        "number of wingers under 22",
        "how many defenders with 1000+ minutes"
    ],
    
    "LIST_QUERIES": [
        "list all Premier League players",
        "show me strikers in La Liga", 
        "all players under 21",
        "list midfielders in Serie A",
        "show goalkeepers with 500+ minutes",
        "all defenders in Bundesliga",
        "list players over 30 in Ligue 1",
        "show wingers under 25"
    ],
    
    "PLAYER_REPORTS": [
        "Pedri report",
        "Messi report", 
        "Haaland report",
        "Mbappe report",
        "Cristiano Ronaldo report",
        "Benzema report",
        "Modric report",
        "Salah report",
        "Kane report",
        "Vinicius report"
    ],
    
    "STAT_DEFINITIONS": [
        "define xG",
        "what is xA per 90",
        "explain Goals + Assists per 90",
        "what does npxG mean",
        "xG definition",
        "Poss+/- meaning",
        "how is xA calculated",
        "define progressive passes per 90",
        "what is clean sheets",
        "explain save percentage",
        "define tackles per 90",
        "what does dribble success rate mean"
    ],
    
    "COMPLEX_COMBINATIONS": [
        "top 5 strikers under 25 in Premier League with 1000+ minutes by goals per 90",
        "how many midfielders over 28 in La Liga with at least 2 assists per 90",
        "best 3 defenders under 30 in top 5 leagues with more than 8 tackles per 90",
        "top 10 players by xG + xA under 23 in Premier League and La Liga",
        "strikers in Serie A and Bundesliga under 26 with 500+ minutes by npxG",
        "midfielders over 25 in top 5 leagues with more than 5 key passes per 90",
        "wingers under 24 in Premier League and La Liga by goals + assists",
        "defenders over 30 with 1500+ minutes by aerial duels won"
    ],
    
    "GOALKEEPER_SPECIFIC": [
        "top 5 goalkeepers by saves per 90",
        "best keepers by clean sheets",
        "highest save percentage in Premier League",
        "goalkeepers with most saves in La Liga",
        "top keepers by xG conceded per 90",
        "best goalkeepers under 25",
        "highest saves in top 5 leagues",
        "keepers with 500+ minutes by save percentage"
    ],
    
    "DEFENSIVE_STATS": [
        "top defenders by tackles per 90",
        "best players by interceptions per 90",
        "highest blocks per 90",
        "most clearances in Premier League",
        "top centre-backs by aerial duels won",
        "best full-backs by tackles + interceptions",
        "defenders with most duels won",
        "highest defensive actions per 90"
    ],
    
    "PASSING_STATS": [
        "top players by passes per 90",
        "best passers by pass completion %",
        "highest key passes per 90",
        "most progressive passes in Serie A",
        "top midfielders by through passes",
        "best players by long passes per 90",
        "highest cross accuracy in Premier League",
        "most deep completions per 90"
    ],
    
    "EDGE_CASES_TYPOS": [
        "top 5 players by goles per 90",  # misspelled goals
        "best assits in Premier League",  # misspelled assists
        "Serei A top players",  # misspelled Serie A
        "hihgest xG per 90",  # misspelled highest
        "Premire League strikers",  # misspelled Premier
        "La Lgaa best players",  # misspelled La Liga
        "top players by gaols",  # misspelled goals
        "best asists per 90"  # misspelled assists
    ],
    
    "EDGE_CASES_INVALID": [
        "top 5 players in Fake League",  # non-existent league
        "Fake Player report",  # non-existent player
        "define fake statistic",  # non-existent stat
        "best players in Mars League",  # fictional league
        "Superman report",  # fictional player
        "top players by magic points",  # fictional stat
        "highest unicorn goals",  # nonsense stat
        "best players in season 2050"  # future reference
    ],
    
    "EDGE_CASES_AMBIGUOUS": [
        "top players",  # no stat specified
        "Premier League players",  # no stat specified
        "best strikers",  # no stat specified
        "show me players",  # too vague
        "La Liga",  # just league name
        "under 23",  # just age filter
        "with 1000+ minutes",  # just minutes filter
        "football players"  # too general
    ],
    
    "NATURAL_LANGUAGE_VARIATIONS": [
        "who are the best goalscorers in Premier League",
        "show me the top assist providers in La Liga",
        "which players have the highest expected goals",
        "find me strikers with most goals per game",
        "I want to see the best passers in Serie A",
        "can you show defenders with most tackles",
        "what players score the most goals",
        "who provides the most assists per match"
    ],
    
    "STRESS_TESTS": [
        "top 100 players by goals per 90 in all leagues",
        "every player under 18 with 2000+ minutes",
        "all goalkeepers with 100% save percentage",
        "players with exactly 0.123456 xG per 90",
        "top 1000 players by assists in Premier League",
        "strikers with negative goals per 90",
        "players older than 50 years",
        "best players with 0 minutes played"
    ]
}

def save_results_to_json(all_results, stats, filename=None):
    """Save test results to a JSON file with timestamp and detailed information."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
    
    # Create comprehensive results structure
    json_output = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(all_results),
            "successful_tests": len([r for r in all_results if r.get("success", False)]),
            "failed_tests": len([r for r in all_results if not r.get("success", False)]),
            "success_rate": round(len([r for r in all_results if r.get("success", False)]) / len(all_results) * 100, 1) if all_results else 0,
            "average_response_time": round(sum(r.get("response_time", 0) or 0 for r in all_results) / len(all_results), 3) if all_results else 0
        },
        "category_performance": stats.get("category_performance", {}),
        "query_type_distribution": stats.get("query_type_distribution", {}),
        "error_analysis": stats.get("error_analysis", {}),
        "detailed_results": []
    }
    
    # Add detailed results for each test
    for result in all_results:
        detailed_result = {
            "query": result.get("query", ""),
            "category": result.get("category", "UNKNOWN"),
            "success": result.get("success", False),
            "response_time": result.get("response_time"),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
            "response": result.get("response"),
            "preprocessed": result.get("preprocessed"),
            "data_analysis": result.get("data_analysis")
        }
        json_output["detailed_results"].append(detailed_result)
    
    # Separate successful and failed tests
    json_output["successful_tests"] = [r for r in json_output["detailed_results"] if r["success"]]
    json_output["failed_tests"] = [r for r in json_output["detailed_results"] if not r["success"]]
    
    # Add slow queries (>2 seconds)
    json_output["slow_queries"] = [
        r for r in json_output["detailed_results"] 
        if r.get("response_time") is not None and r.get("response_time", 0) > 2.0
    ]
    
    # Save to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📄 Test results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        return None

def test_query(query: str, category: str) -> dict:
    """Test a single query and return comprehensive results."""
    try:
        start_time = time.time()
        response = requests.post(API_URL, json={"message": query, "history": []})
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if the response contains an error in the summary or message
            summary = data.get("summary", "")
            message = data.get("message", "")
            
            # Define error indicators that suggest the API encountered an error
            error_indicators = [
                "An error occurred",
                "Error:",
                "Exception:",
                "Failed to",
                "Could not",
                "Unable to",
                "Invalid",
                "Not found",
                "nothing to repeat at position",
                "Traceback",
                "ValueError:",
                "TypeError:",
                "KeyError:",
                "AttributeError:",
                "IndexError:",
                "NameError:",
                "SyntaxError:"
            ]
            
            # Check if summary or message contains error indicators
            is_error_response = False
            error_detail = None
            
            for indicator in error_indicators:
                if indicator.lower() in summary.lower() or indicator.lower() in message.lower():
                    is_error_response = True
                    error_detail = summary if indicator.lower() in summary.lower() else message
                    break
            
            # Also check if essential fields are missing (could indicate a malformed response)
            if not is_error_response and not data.get("table") and not summary and not message:
                is_error_response = True
                error_detail = "Empty or malformed response - no data returned"
            
            if is_error_response:
                return {
                    "query": query,
                    "category": category,
                    "status": "API_ERROR",
                    "success": False,
                    "response_time": round(response_time, 3),
                    "error": error_detail[:150],
                    "error_type": "API_ERROR",
                    "error_message": error_detail,
                    "response": data,
                    "preprocessed": data.get("preprocessed", {}),
                    "data_analysis": data.get("data_analysis", {})
                }
            else:
                return {
                    "query": query,
                    "category": category,
                    "status": "SUCCESS",
                    "success": True,
                    "response_time": round(response_time, 3),
                    "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                    "has_table": bool(data.get("table")),
                    "query_type": data.get("preprocessed", {}).get("query_type", "UNKNOWN"),
                    "filters_applied": data.get("data_analysis", {}).get("applied_filters", []),
                    "player_count": data.get("data_analysis", {}).get("count", 0),
                    "error": None,
                    "raw_response_size": len(json.dumps(data)),
                    "response": data,
                    "preprocessed": data.get("preprocessed", {}),
                    "data_analysis": data.get("data_analysis", {})
                }
        else:
            return {
                "query": query,
                "category": category,
                "status": "HTTP_ERROR",
                "success": False,
                "response_time": round(response_time, 3),
                "error": f"HTTP {response.status_code}: {response.text[:100]}",
                "error_type": "HTTP_ERROR",
                "error_message": f"HTTP {response.status_code}: {response.text[:100]}"
            }
    except Exception as e:
        return {
            "query": query,
            "category": category,
            "status": "EXCEPTION",
            "success": False,
            "response_time": None,
            "error": str(e)[:150],
            "error_type": "EXCEPTION",
            "error_message": str(e)[:150]
        }

def calculate_test_statistics(all_results):
    """Calculate comprehensive test statistics from results."""
    
    if not all_results:
        return {}
    
    # Basic stats
    total_tests = len(all_results)
    successful_tests = len([r for r in all_results if r.get("success", False)])
    failed_tests = total_tests - successful_tests
    success_rate = round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0
    
    # Response time stats
    response_times = [r.get("response_time", 0) or 0 for r in all_results]
    avg_response_time = round(sum(response_times) / len(response_times), 2) if response_times else 0
    min_response_time = min(response_times) if response_times else 0
    max_response_time = max(response_times) if response_times else 0
    
    # Category performance
    category_performance = {}
    categories = set(r.get("category", "UNKNOWN") for r in all_results)
    
    for category in categories:
        category_results = [r for r in all_results if r.get("category") == category]
        category_successful = len([r for r in category_results if r.get("success", False)])
        category_total = len(category_results)
        category_success_rate = round(category_successful / category_total * 100, 1) if category_total > 0 else 0
        category_avg_time = round(sum(r.get("response_time", 0) or 0 for r in category_results) / category_total, 2) if category_total > 0 else 0
        
        category_performance[category] = {
            "successful": category_successful,
            "total": category_total,
            "success_rate": category_success_rate,
            "avg_response_time": category_avg_time
        }
    
    # Query type distribution
    query_type_distribution = {}
    for result in all_results:
        preprocessed = result.get("preprocessed", {})
        query_type = preprocessed.get("query_type", "UNKNOWN") if isinstance(preprocessed, dict) else "UNKNOWN"
        query_type_distribution[query_type] = query_type_distribution.get(query_type, 0) + 1
    
    # Error analysis
    error_analysis = {}
    failed_results = [r for r in all_results if not r.get("success", False)]
    for result in failed_results:
        error_type = result.get("error_type", "UNKNOWN")
        error_analysis[error_type] = error_analysis.get(error_type, 0) + 1
    
    return {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "category_performance": category_performance,
        "query_type_distribution": query_type_distribution,
        "error_analysis": error_analysis
    }

def run_comprehensive_tests(base_url="http://localhost:8000", save_json=True, json_filename=None):
    """Run all test cases with comprehensive reporting."""
    print("🚀 Starting Enhanced Comprehensive Test Suite")
    print("=" * 80)
    
    # Prepare test categories
    categories_to_test = dict(test_cases)
    
    all_results = []
    category_stats = {}
    total_queries = sum(len(queries) for queries in categories_to_test.values())
    
    print(f"📊 Total test queries: {total_queries}")
    print(f"📁 Categories: {len(categories_to_test)}")
    
    # Create list of all (query, category) pairs
    all_test_queries = []
    for category, queries in categories_to_test.items():
        for query in queries:
            all_test_queries.append((query, category))
    
    # Run tests
    for i, (query, category) in enumerate(all_test_queries, 1):
        print(f"\n[{i:3d}/{total_queries}] 🧪 {category}: '{query}'")
        
        result = test_query(query, category)
        all_results.append(result)
        
        # Print result
        if result["status"] == "SUCCESS":
            response_time = result.get("response_time", 0)
            player_count = result.get("player_count", 0)
            print(f"    ✅ {result['query_type']} | {response_time:.2f}s | {player_count} players")
            print(f"    💬 {result['summary'][:80]}...")
        elif result["status"] == "API_ERROR":
            response_time = result.get("response_time", 0)
            print(f"    🚨 API_ERROR | {response_time:.2f}s")
            print(f"    💬 {result.get('error', 'Unknown API error')[:80]}...")
        else:
            print(f"    ❌ {result['status']}: {result.get('error', 'Unknown error')[:60]}...")
        
        # Adaptive delay based on response time
        if result.get("response_time"):
            delay = max(0.1, min(0.5, result["response_time"] / 2))
        else:
            delay = 0.2
        time.sleep(delay)
    
    # Calculate statistics
    for category in categories_to_test.keys():
        category_results = [r for r in all_results if r["category"] == category]
        success_count = sum(1 for r in category_results if r["status"] == "SUCCESS")
        
        avg_response_time = None
        response_times = [r["response_time"] for r in category_results if r["response_time"]]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
        
        category_stats[category] = {
            "total": len(category_results),
            "success": success_count,
            "failed": len(category_results) - success_count,
            "success_rate": (success_count / len(category_results)) * 100,
            "avg_response_time": round(avg_response_time, 3) if avg_response_time else None
        }
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    # Overall stats
    total_tests = len(all_results)
    total_success = sum(1 for r in all_results if r["status"] == "SUCCESS")
    total_failed = total_tests - total_success
    overall_success_rate = (total_success / total_tests) * 100
    
    print(f"🔢 Total Tests: {total_tests}")
    print(f"✅ Successful: {total_success} ({overall_success_rate:.1f}%)")
    print(f"❌ Failed: {total_failed} ({100-overall_success_rate:.1f}%)")
    
    # Response time stats
    response_times = [r["response_time"] for r in all_results if r["response_time"]]
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        print(f"⏱️  Avg Response Time: {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")
    
    # Category breakdown
    print(f"\n📁 Category Performance:")
    for category, stats in sorted(category_stats.items()):
        status_icon = "✅" if stats["success_rate"] >= 90 else "⚠️" if stats["success_rate"] >= 70 else "❌"
        time_info = f" | {stats['avg_response_time']:.2f}s" if stats["avg_response_time"] else ""
        print(f"  {status_icon} {category:25s}: {stats['success']:2d}/{stats['total']} ({stats['success_rate']:5.1f}%){time_info}")
    
    # Query type distribution
    query_types = {}
    for result in all_results:
        if result["status"] == "SUCCESS":
            qtype = result.get("query_type", "UNKNOWN")
            query_types[qtype] = query_types.get(qtype, 0) + 1
    
    print(f"\n🏷️  Query Type Distribution:")
    for qtype, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_success) * 100
        print(f"  {qtype:15s}: {count:3d} queries ({percentage:4.1f}%)")
    
    # Error analysis
    error_types = {}
    for result in all_results:
        if result["status"] != "SUCCESS":
            error_type = result["status"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if error_types:
        print(f"\n❌ Error Analysis:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_failed) * 100 if total_failed > 0 else 0
            print(f"  {error_type:15s}: {count:3d} errors ({percentage:4.1f}%)")
    
    # Sample failures
    failed_queries = [r for r in all_results if r["status"] != "SUCCESS"]
    if failed_queries:
        print(f"\n🔍 Sample Failed Queries:")
        for result in failed_queries[:10]:  # Show first 10 failures
            print(f"  • [{result['category']}] '{result['query'][:60]}...'")
            print(f"    {result['status']}: {result.get('error', 'Unknown')[:80]}")
    
    # Performance insights
    slow_queries = [r for r in all_results if r.get("response_time") is not None and r.get("response_time", 0) > 2.0]
    if slow_queries:
        print(f"\n🐌 Slow Queries (>2s):")
        for result in sorted(slow_queries, key=lambda x: x.get("response_time", 0), reverse=True)[:5]:
            print(f"  • {result['response_time']:.2f}s: '{result['query'][:60]}...'")
    
    print(f"\n🎉 Test Suite Complete!")
    stats = calculate_test_statistics(all_results)
    
    # Save to JSON if requested
    json_filename_used = None
    if save_json:
        json_filename_used = save_results_to_json(all_results, stats, json_filename)
    
    return all_results, stats, json_filename_used

def run_quick_smoke_test():
    """Run a quick smoke test with essential queries."""
    essential_queries = [
        ("top 5 goals per 90", "SMOKE"),
        ("Premier League strikers", "SMOKE"),
        ("Messi report", "SMOKE"),
        ("define xG", "SMOKE"),
        ("players with 500+ minutes", "SMOKE")
    ]
    
    print("💨 Quick Smoke Test")
    print("-" * 30)
    
    for query, category in essential_queries:
        result = test_query(query, category)
        status = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"{status} {query}")
    
    print("💨 Smoke test complete!")

if __name__ == "__main__":
    print("🚀 Starting comprehensive football metrics tests...")
    
    try:
        results, stats, json_file = run_comprehensive_tests(
            base_url="http://localhost:8000",
            save_json=True  # Enable JSON output
        )
        
        print(f"\n🎉 Testing completed!")
        if json_file:
            print(f"📊 Detailed results saved to: {json_file}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")
        import traceback
        traceback.print_exc()