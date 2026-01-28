#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

/**
 * Lightweight test harness for CUDA kernel verification.
 *
 * Usage:
 *   TestHarness harness("Dot Product Tests");
 *   harness.run("single element", []() { return test_single_element(); });
 *   harness.run("all zeros", []() { return test_all_zeros(); });
 *   return harness.summarize();  // Returns 0 on success, 1 on failure
 */
class TestHarness {
public:
    explicit TestHarness(const std::string& suiteName)
        : suiteName_(suiteName), passed_(0), failed_(0) {
        printf("\n=== %s ===\n", suiteName_.c_str());
    }

    /**
     * Run a single test case.
     * @param name Test name for display
     * @param testFn Function that returns true on pass, false on fail
     */
    void run(const std::string& name, std::function<bool()> testFn) {
        bool result = false;
        try {
            result = testFn();
        } catch (const std::exception& e) {
            printf("  [FAIL] %s - Exception: %s\n", name.c_str(), e.what());
            failed_++;
            failedTests_.push_back(name);
            return;
        } catch (...) {
            printf("  [FAIL] %s - Unknown exception\n", name.c_str());
            failed_++;
            failedTests_.push_back(name);
            return;
        }

        if (result) {
            printf("  [PASS] %s\n", name.c_str());
            passed_++;
        } else {
            printf("  [FAIL] %s\n", name.c_str());
            failed_++;
            failedTests_.push_back(name);
        }
    }

    /**
     * Print summary and return exit code.
     * @return 0 if all tests passed, 1 if any failed
     */
    int summarize() const {
        printf("\n--- %s Summary ---\n", suiteName_.c_str());
        printf("Passed: %d, Failed: %d, Total: %d\n",
               passed_, failed_, passed_ + failed_);

        if (!failedTests_.empty()) {
            printf("\nFailed tests:\n");
            for (const auto& name : failedTests_) {
                printf("  - %s\n", name.c_str());
            }
        }

        return failed_ == 0 ? 0 : 1;
    }

    int passed() const { return passed_; }
    int failed() const { return failed_; }
    bool allPassed() const { return failed_ == 0; }

private:
    std::string suiteName_;
    int passed_;
    int failed_;
    std::vector<std::string> failedTests_;
};

#endif // TEST_HARNESS_H
