// go-counting logs a counter once a second. Built with Go 1.25 or later (go.mod), whose runtime
// keeps the cgroup's CPU-limit file open to size GOMAXPROCS: a restore under a new cgroup has to
// cope with that file.
package main

import (
	"fmt"
	"os"
	"runtime"
	"time"
)

func main() {
	fmt.Printf("go-counting: %s, GOMAXPROCS %d, GODEBUG=%q\n", runtime.Version(), runtime.GOMAXPROCS(0), os.Getenv("GODEBUG"))
	for i := 0; ; i++ {
		fmt.Printf("%d %s\n", i, time.Now().Format(time.RFC3339))
		time.Sleep(time.Second)
	}
}
