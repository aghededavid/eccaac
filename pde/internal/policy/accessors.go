// ECCAAC — Policy Decision Engine
// Store accessor methods — used by gRPC server and REST API.

package policy

import "github.com/eccaac/pde/internal/types"

// GetStore returns the policy store (used by gRPC server and REST API).
func (e *Engine) GetStore() *Store { return e.store }

// GetGroup returns a policy group by ID.
func (s *Store) GetGroup(id string) (types.PolicyGroup, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	g, ok := s.groups[id]
	return g, ok
}
