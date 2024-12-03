package scheduler

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func createGPUNode(name string, tflops, vram string) *tfv1.GPUNode {
	return &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Status: tfv1.GPUNodeStatus{
			Available: tfv1.Resource{
				Tflops: resource.MustParse(tflops),
				Vram:   resource.MustParse(vram),
			},
		},
	}
}

func createRequest(tflops, vram string) tfv1.Resource {
	return tfv1.Resource{
		Tflops: resource.MustParse(tflops),
		Vram:   resource.MustParse(vram),
	}
}

func TestNaiveScheduler_Schedule(t *testing.T) {
	tests := []struct {
		name                string
		nodes               []*tfv1.GPUNode
		request             tfv1.Resource
		wantNode            string
		wantError           bool
		wantRemainingTflops string
		wantRemainingVram   string
	}{
		{
			name: "simple match",
			nodes: []*tfv1.GPUNode{
				createGPUNode("node1", "100", "16Gi"),
			},
			request:             createRequest("50", "8Gi"),
			wantNode:            "node1",
			wantError:           false,
			wantRemainingTflops: "50",
			wantRemainingVram:   "8Gi",
		},
		{
			name:      "no nodes",
			nodes:     []*tfv1.GPUNode{},
			request:   createRequest("50", "8Gi"),
			wantNode:  "",
			wantError: true,
		},
		{
			name: "insufficient resources",
			nodes: []*tfv1.GPUNode{
				createGPUNode("node1", "40", "16Gi"),
			},
			request:   createRequest("50", "8Gi"),
			wantNode:  "",
			wantError: true,
		},
		{
			name: "multiple nodes, first fit",
			nodes: []*tfv1.GPUNode{
				createGPUNode("node1", "40", "16Gi"),
				createGPUNode("node2", "100", "32Gi"),
			},
			request:             createRequest("50", "8Gi"),
			wantNode:            "node2",
			wantError:           false,
			wantRemainingTflops: "50",
			wantRemainingVram:   "24Gi",
		},
		{
			name: "exact match",
			nodes: []*tfv1.GPUNode{
				createGPUNode("node1", "50", "8Gi"),
			},
			request:             createRequest("50", "8Gi"),
			wantNode:            "node1",
			wantError:           false,
			wantRemainingTflops: "0",
			wantRemainingVram:   "0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewNaiveScheduler()

			// Add nodes
			for _, node := range tt.nodes {
				s.OnAdd(node)
			}

			// Try to schedule
			got, err := s.Schedule(tt.request)

			// Check error
			if (err != nil) != tt.wantError {
				t.Errorf("Schedule() error = %v, wantError %v", err, tt.wantError)
				return
			}

			// Check result
			if !tt.wantError {
				if got == nil {
					t.Error("Schedule() returned nil node when error not expected")
					return
				}
				if got.Name != tt.wantNode {
					t.Errorf("Schedule() got node = %v, want %v", got.Name, tt.wantNode)
				}

				// Check remaining resources
				if tt.wantRemainingTflops != "" {
					wantTflops := resource.MustParse(tt.wantRemainingTflops)
					if got.Status.Available.Tflops.Cmp(wantTflops) != 0 {
						t.Errorf("Remaining Tflops = %v, want %v", got.Status.Available.Tflops.String(), tt.wantRemainingTflops)
					}
				}
				if tt.wantRemainingVram != "" {
					wantVram := resource.MustParse(tt.wantRemainingVram)
					if got.Status.Available.Vram.Cmp(wantVram) != 0 {
						t.Errorf("Remaining Vram = %v, want %v", got.Status.Available.Vram.String(), tt.wantRemainingVram)
					}
				}
			}
		})
	}
}

func TestNaiveScheduler_NodeOperations(t *testing.T) {
	s := NewNaiveScheduler()
	node1 := createGPUNode("node1", "100", "16Gi")
	request := createRequest("50", "8Gi")

	// Test OnAdd
	s.OnAdd(node1)
	got, err := s.Schedule(request)
	if err != nil || got.Name != "node1" {
		t.Errorf("After OnAdd: Schedule() got = %v, want node1", got)
	}

	// Test OnUpdate
	node1Updated := createGPUNode("node1", "40", "16Gi")
	s.OnUpdate(node1, node1Updated)
	_, err = s.Schedule(request)
	if err == nil {
		t.Error("After OnUpdate: Schedule() should fail with insufficient resources")
	}

	// Test OnDelete
	s.OnDelete(node1Updated)
	_, err = s.Schedule(request)
	if err == nil {
		t.Error("After OnDelete: Schedule() should fail with no nodes")
	}
}
