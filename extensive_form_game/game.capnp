using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("game_schema");

@0xf1f28acd82d7ca07;

struct Infoset {
  startSequenceId @0 :UInt32;
  endSequenceId @1 :UInt32;
  parentSequenceId @2 :UInt32;
}

struct Treeplex {
  infosets @0 :List(Infoset);
}

struct PayoffMatrix {
  entries @0 :List(Entry);

  struct Entry {
    sequences @0 :List(UInt32);
    payoffs @1 :List(Float64);
    chanceFactor @2 :Float64;
  }
}

struct Game {
  treeplexes @0 :List(Treeplex);
  payoffMatrix @1 :PayoffMatrix;

  notes @2 :Text;
}

