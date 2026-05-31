// Genealogy graph schema — constraints (index-backed) and indexes.
// Apply with: cypher-shell -f src/genealogy_rag/graph/schema.cypher
// (Neo4jGraphStore.ensure_schema() runs the equivalent at ingest time.)

// --- uniqueness constraints (each backs a range index → O(1) id seeks) ---
CREATE CONSTRAINT person_id   IF NOT EXISTS FOR (p:Person)   REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;

// --- secondary index for name-based entity linking from the query side ---
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);

// --- node model ---
//   (:Person   {id, name})
//   (:Document {id})              // text/citation live in the document store, not the graph
// --- relationship model ---
//   (child:Person)-[:CHILD_OF]->(parent:Person)
//   (parent:Person)-[:PARENT_OF]->(child:Person)
//   (a:Person)-[:SPOUSE_OF]->(b:Person)        // materialised both directions
//   (p:Person)-[:MENTIONS]->(d:Document)       // provenance edge

// example: maternal grandfather of a person (the multi-hop query graph-RAG exploits)
//   MATCH (g4:Person {name:$name})-[:CHILD_OF]->(mother:Person)
//   WHERE NOT (mother)-[:SPOUSE_OF]->(:Person)-[:PARENT_OF]->(g4)  // pick mother
//   MATCH (mother)-[:CHILD_OF]->(gf:Person) WHERE gf.sex = 'M'
//   RETURN gf.name;
