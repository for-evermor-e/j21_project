#![allow(warnings)]

extern crate pest;
#[macro_use]
extern crate pest_derive;

use pest::Parser;
use pest::prec_climber::{Assoc, Operator, PrecClimber};  // operator precedence
use std::fs;
use std::collections::HashMap;


//============================================ ASTNode ============================================//

#[derive(Debug)]
#[derive(Copy, Clone)]
pub enum VecType {
    int,
    float
}

#[derive(Debug)]
#[derive(Copy, Clone)]
#[derive(PartialEq)]
pub enum PortType {
    weight,
    input,
    bias,
    output
}

#[derive(Debug)]
#[derive(Copy, Clone)]
pub enum Op {
    mult,
    add
}

#[derive(Debug)]
#[derive(Clone)]
pub enum ASTNode {
    Vector {
        vec_type: VecType,
        dim0: i8,
        dim1: i8
    },

    MAC {
        weight: String,
        input: String,
        bias: String
    },

    Decl {
        port_type: PortType,
        name: String,
        vec: Box<ASTNode>  // vector(without Box, compiler fails)
    },
    
    Wire {
        port_type: PortType,
        name: String,
        mac: Box<ASTNode>  // MAC(without Box, compiler fails)
    }
}

impl ASTNode {
    fn port_type(&self) -> PortType {  // return ASTNode's port_type: weight, input, bias or output
        match self {
            ASTNode::Decl { port_type, .. } => *port_type,
            ASTNode::Wire { port_type, .. } => *port_type,
            _ => unreachable!()
        }
    }

    fn name(&self) -> String {
        match self {
            ASTNode::Decl { name, .. } => name.clone(),
            ASTNode::Wire { name, .. } => name.clone(),
            _ => unreachable!()
        }
    }

    fn Vector(&self) -> ASTNode {
        match self {
            ASTNode::Decl { vec, .. } => (**vec).clone(),
            _ => unreachable!()
        }
    }

    fn vec_type(&self) -> VecType {
        match self {
            ASTNode::Vector { vec_type, .. } => *vec_type,
            _ => unreachable!()
        }
    }

    fn dim0(&self) -> i8 {
        match self {
            ASTNode::Vector { dim0, .. } => *dim0,
            _ => unreachable!()
        }
    }

    fn dim1(&self) -> i8 {
        match self {
            ASTNode::Vector { dim1, .. } => *dim1,
            _ => unreachable!()
        }
    }

    fn MAC(&self) -> ASTNode {
        match self {
            ASTNode::Wire { mac, .. } => (**mac).clone(),
            _ => unreachable!()
        } 
    }

    fn weight(&self) -> String {
        match self {
            ASTNode::MAC { weight, .. } => weight.clone(),
            _ => unreachable!()
        }
    }

    fn input(&self) -> String {
        match self {
            ASTNode::MAC { input, .. } => input.clone(),
            _ => unreachable!()
        }
    }

    fn bias(&self) -> String {
        match self {
            ASTNode::MAC { bias, .. } => bias.clone(),
            _ => unreachable!()
        }
    }
}

//============================================ ASTNode ============================================//


//============================================ Parser =============================================//

#[derive(Debug)]
#[derive(Clone)]
pub struct AboutID {
    port_type: PortType,
    vec: ASTNode
}


#[derive(Parser)]
#[grammar = "grammar.pest"]
struct J21Parser;

pub fn parse(file: &str, ID_HashMap: &mut HashMap<String, AboutID>) -> Vec<ASTNode> {  // parsing by recursive call
    let mut ASTNodes: Vec<ASTNode> = Vec::new();             // ASTNode를 enum이 아닌 struct로 정의하면 vector를 만들기 어려움
    
    let forToken = J21Parser::parse(Rule::program, file);
    
    // Printing Tokens
    println!("\nPrinting Tokens:\n");
    let tokens = forToken.unwrap().tokens();
    for token in tokens {
        println!("{:?}", token);
    }                                                     

    let program = J21Parser::parse(Rule::program, file);     // .?를 쓰려면 return type이 Result<Vec<ASTNode>, Error<Rule>>

    let pairs = program.unwrap();

    for pair in pairs {  // parsing
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::decl => ASTNodes.push(parse_Decl(inner_pair, ID_HashMap)),
                Rule::wire => ASTNodes.push(parse_Wire(inner_pair, ID_HashMap)),
                Rule::EOI => (),
                _ => unreachable!()
            };
        }
    }

    // Printing all elements in ID_HashMap
    println!("\n\nPrinting all elements in ID_HashMap:\n");
    println!("{:?}", ID_HashMap);
    for val in ID_HashMap.keys() {
        println!("{:?}", val);
    }

    return ASTNodes;
}

// vector = ${ vec_type ~ "[" ~ WHITESPACE* ~ int ~ WHITESPACE* ~ "," ~ WHITESPACE* ~ int ~ WHITESPACE* ~ "]" }
fn parse_Vector(pair: pest::iterators::Pair<Rule>) -> ASTNode {  // call parse_Vector() inside parse_Decl()
    let mut pair = pair.into_inner();

    let vec_type = match pair.next().unwrap().as_str() {
        "int" => VecType::int,
        "float" => VecType::float,
        _ => panic!("Wrong Vector Type! It should be 'int' or 'float'.")
    };

    let dim0 = pair.next().unwrap().as_str().parse::<i8>().unwrap();

    let dim1 = pair.next().unwrap().as_str().parse::<i8>().unwrap();
    
    return ASTNode::Vector{
        vec_type: vec_type,
        dim0: dim0,
        dim1: dim1
    };
}

// mac = ${ "MAC" ~ "(" ~ WHITESPACE* ~ id ~ WHITESPACE* ~ "," ~ WHITESPACE* ~ id ~ WHITESPACE* ~ "," ~ WHITESPACE* ~ id ~ WHITESPACE* ~ ")" }
fn parse_MAC(pair: pest::iterators::Pair<Rule>, ID_HashMap: &HashMap<String, AboutID>) -> ASTNode {
                                                // MAC ASTNode에는 id들만 있고 calyx generator에서 HashMap을 통해 검색
    let mut pair = pair.into_inner();
    
    let weight = pair.next().unwrap().as_str().to_string();
    
    if ID_HashMap.get(&weight).unwrap().port_type != PortType::weight {
        panic!("Arguments of MAC should be order of weight, input, and bias!");
    }

    let input = pair.next().unwrap().as_str().to_string();

    if ID_HashMap.get(&input).unwrap().port_type != PortType::input {
        panic!("Arguments of MAC should be order of weight, input, and bias!");
    }
    
    let bias = pair.next().unwrap().as_str().to_string();
    
    if ID_HashMap.get(&bias).unwrap().port_type != PortType::bias {
        panic!("Arguments of MAC should be order of weight, input, and bias!");
    }

    if ID_HashMap.get(&weight).unwrap().vec.dim1() != ID_HashMap.get(&input).unwrap().vec.dim0() {
        panic!("dim1 of weight should be the same as dim0 of input!");
    }

    if ID_HashMap.get(&weight).unwrap().vec.dim0() != ID_HashMap.get(&bias).unwrap().vec.dim0() {
        panic!("dim0 of weight should be the same as dim0 of bias!");
    }

    if ID_HashMap.get(&input).unwrap().vec.dim1() != ID_HashMap.get(&bias).unwrap().vec.dim1() {
        panic!("dim1 of input should be the same as dim1 of bias!");
    }

    return ASTNode::MAC{
        weight: weight,
        input: input,
        bias: bias
    };
}

/*fn parse_Expr(pair: pest::iterators::Pair<Rule>) -> ASTNode {
    let mut pair = pair.into_inner();
    
    let vec_1 = pair.next().unwrap().as_str().to_string();
    
    let op_1 = match pair.next().unwrap().as_str() {
        "+" => Op::plus,
        "*" => Op::mul,
        _ => panic!("Wrong Operator! It should be '+' or '*'.")
    };
    
    let vec_2 = pair.next().unwrap().as_str().to_string();
    
    let op_2 = match pair.next().unwrap().as_str() {
        "+" => Op::plus,
        "*" => Op::mul,
        _ => panic!("Wrong Operator! It should be '+' or '*'.")
    };
    
    let vec_3 = pair.next().unwrap().as_str().to_string();

    return ASTNode::Expr{
        vec_1: vec_1,
        op_1: op_1,
        vec_2: vec_2,
        op_2: op_2,
        vec_3: vec_3
    };
}*/

// decl = { port_type ~ id ~ "=" ~ vector ~ ";" }
fn parse_Decl(pair: pest::iterators::Pair<Rule>, ID_HashMap: &mut HashMap<String, AboutID>) -> ASTNode {
    let mut pair = pair.into_inner();

    let port_type = match pair.next().unwrap().as_str() {
        "weight" => PortType::weight,
        "input" => PortType::input,
        "bias" => PortType::bias,
        _ => panic!("Wrong Decl Port Type! It should be one of 'weight', 'input' and 'bias'.")
    };

    let name = pair.next().unwrap().as_str().to_string();
    let name_for_HashMap = name.clone();
    
    let vec_pair = pair.next().unwrap();
    let vec_pair_for_HashMap = vec_pair.clone();

    let vec = Box::new(parse_Vector(vec_pair));  // Decl pair의 inner_pair인 vec_pair로 parse_Vector()
    
    ID_HashMap.entry(name_for_HashMap)
        .or_insert(AboutID{
            port_type: port_type,
            vec: parse_Vector(vec_pair_for_HashMap)
        });

    return ASTNode::Decl{
        port_type: port_type,
        name: name,
        vec: vec
    };
}

// wire = { port_type ~ id ~ "=" ~ mac ~ ";" }
fn parse_Wire(pair: pest::iterators::Pair<Rule>, ID_HashMap: &HashMap<String, AboutID>) -> ASTNode {
    let mut pair = pair.into_inner();

    let port_type = match pair.next().unwrap().as_str() {
        "output" => PortType::output,
        _ => panic!("Wrong Wire Port Type! It should be 'output'.")
    };

    let name = pair.next().unwrap().as_str().to_string();

    let mac_pair = pair.next().unwrap();

    let mac = Box::new(parse_MAC(mac_pair, ID_HashMap));

    return ASTNode::Wire{
        port_type: port_type,
        name: name,
        mac: mac
    };
}

//============================================ Parser =============================================//


//======================================== Calyx Generator ========================================//

fn tap(output_file: &mut String) {
    output_file.push_str("    ");
}

fn Calyx_Generator(output_file: &mut String, ASTNodes: &Vec<ASTNode>, ID_HashMap: &HashMap<String, AboutID>,
                   Mems: &HashMap<String, AboutMem>, Comps: &Vec<Comp>, Regs: &mut HashMap<String, AboutReg>, Adders: &mut HashMap<String, AboutAdder>, LTs: &mut HashMap<String, AboutLT>) {
    generate_Import(output_file);
    generate_Component(output_file, ASTNodes, ID_HashMap, Mems, Comps, Regs, Adders, LTs);
}

fn generate_Import(output_file: &mut String) {
    let stdlib = r#"import "primitives/std.lib";"#;

    output_file.push_str(stdlib);
    output_file.push_str("\n");
}

#[derive(Debug)]
#[derive(Clone)]
pub enum AboutMem {
    Mem1D {
        // parameters
        width: i8,
        size: i8,
        idx_size: i8,

        // I/O ports
        addr0: String,
        write_data: String,
        write_en: String,
        read_data: String,
        done: String
    },
    Mem2D {
        // parameters
        width: i8,
        dim0_size: i8,
        dim1_size: i8,
        idx_size: i8,

        // I/O ports
        addr0: String,
        addr1: String,
        write_data: String,
        write_en: String,
        read_data: String,
        done: String
    }    
}

impl AboutMem {
    fn width(&self) -> i8 {
        match self {
            AboutMem::Mem1D { width, .. } => *width,
            AboutMem::Mem2D { width, .. } => *width,
            _ => unreachable!()
        }
    }
    fn size(&self) -> i8 {
        match self {
            AboutMem::Mem1D { size, .. } => *size,
            _ => unreachable!()
        }
    }
    fn dim0_size(&self) -> i8 {
        match self {
            AboutMem::Mem2D { dim0_size, .. } => *dim0_size,
            _ => unreachable!()
        }
    }
    fn dim1_size(&self) -> i8 {
        match self {
            AboutMem::Mem2D { dim1_size, .. } => *dim1_size,
            _ => unreachable!()
        }
    }
    fn idx_size(&self) -> i8 {
        match self {
            AboutMem::Mem1D { idx_size, .. } => *idx_size,
            AboutMem::Mem2D { idx_size, .. } => *idx_size,
            _ => unreachable!()
        }
    }
    fn addr0(&self) -> String {
        match self {
            AboutMem::Mem1D { addr0, .. } => addr0.clone(),
            AboutMem::Mem2D { addr0, .. } => addr0.clone(),
            _ => unreachable!()
        }
    }
    fn addr1(&self) -> String {
        match self {
            AboutMem::Mem2D { addr1, .. } => addr1.clone(),
            _ => unreachable!()
        }
    }
    fn write_data(&self) -> String {
        match self {
            AboutMem::Mem1D { write_data, .. } => write_data.clone(),
            AboutMem::Mem2D { write_data, .. } => write_data.clone(),
            _ => unreachable!()
        }
    }
    fn write_en(&self) -> String {
        match self {
            AboutMem::Mem1D { write_en, .. } => write_en.clone(),
            AboutMem::Mem2D { write_en, .. } => write_en.clone(),
            _ => unreachable!()
        }
    }
    fn read_data(&self) -> String {
        match self {
            AboutMem::Mem1D { read_data, .. } => read_data.clone(),
            AboutMem::Mem2D { read_data, .. } => read_data.clone(),
            _ => unreachable!()
        }
    }
    fn done(&self) -> String {
        match self {
            AboutMem::Mem1D { done, .. } => done.clone(),
            AboutMem::Mem2D { done, .. } => done.clone(),
            _ => unreachable!()
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Comp {
    op: Op,
    left: String,  // mul_out = plus_left
    right: String,
    out: String
}

fn generate_Component(output_file: &mut String, ASTNodes: &Vec<ASTNode>, ID_HashMap: &HashMap<String, AboutID>,
                      Mems: &HashMap<String, AboutMem>, Comps: &Vec<Comp>, Regs: &mut HashMap<String, AboutReg>, Adders: &mut HashMap<String, AboutAdder>, LTs: &mut HashMap<String, AboutLT>) {
    
    let component = String::from("component main() -> () {\n");
    
    output_file.push_str(&component);

    generate_Cells(output_file, ID_HashMap, Mems, Comps, Regs, Adders, LTs);
    generate_Wires(output_file, ASTNodes, ID_HashMap, Mems, Comps, Regs, Adders, LTs);
    generate_Control(output_file, ASTNodes);

    output_file.push_str("}");
}

// struct로 mem을 정의해서 parameter, I/O port 저장
// Decl Node -> collect_Mem()을 통해 weight, input, bias의 dimension에 맞는 std_mem 선언
// AboutMem에 port_type 필요 없음(parse_MAC()에서 dimension 안 맞으면 panic!())
fn collect_Mem(Mems: &mut HashMap<String, AboutMem>, name: &String, dim: i8, size: Option<i8>, vec: &Option<ASTNode>) {
    let mem = match dim {
        1 => AboutMem::Mem1D {
            width: 32,
            size: size.unwrap(),
            idx_size: 32,

            addr0: name.clone() + &String::from(".addr0"),
            write_data: name.clone() + &String::from(".write_data"),
            write_en: name.clone() + &String::from(".write_en"),
            read_data: name.clone() + &String::from(".read_data"),
            done: name.clone() + &String::from(".done")
        },
        2 => AboutMem::Mem2D {
            width: 32,
            dim0_size: vec.as_ref().unwrap().dim0(),
            dim1_size: vec.as_ref().unwrap().dim1(),
            idx_size: 32,

            addr0: name.clone() + &String::from(".addr0"),
            addr1: name.clone() + &String::from(".addr1"),
            write_data: name.clone() + &String::from(".write_data"),
            write_en: name.clone() + &String::from(".write_en"),
            read_data: name.clone() + &String::from(".read_data"),
            done: name.clone() + &String::from(".done")
        },
        _ => unreachable!()
    };

    Mems.entry(String::from(name)).or_insert(mem);
}

// Wire Node -> MAC ASTNode를 input으로 받아 MAC()의 3가지 argument vector가 adder/multiplier에 어떻게 연결되는지 저장
fn collect_Comp(Comps: &mut Vec<Comp>, name_M: &String, name_y: &String, mac: &ASTNode) {
    let mult = Comp {
        op: Op::mult,
        left: mac.weight(),
        right: mac.input(),
        out: name_M.to_string()
    };
    
    let add = Comp {
        op: Op::add,
        left: mult.out.clone(),
        right: mac.bias(),
        out: name_y.to_string()
    };

    Comps.push(mult);
    Comps.push(add);
}

#[derive(Debug)]
#[derive(Clone)]
pub struct AboutReg {
    // parameters
    width: i8,

    // I/O ports
    in_: String,
    write_en: String,
    out: String,
    done: String
}

#[derive(Debug)]
#[derive(Clone)]
pub struct AboutAdder {
    // parameters
    width: i8,

    // I/O ports
    left: String,
    right: String,
    out: String
}

#[derive(Debug)]
#[derive(Clone)]
pub struct AboutLT {
    // parameters
    width: i8,

    // I/O ports
    left: String,
    right: String,
    out: String
}

// Mems, Comps를 iter하며 std_mem, std_reg, std_add, std_lt, std_mult_pipe 선언
fn generate_Cells(output_file: &mut String, ID_HashMap: &HashMap<String, AboutID>,
                  Mems: &HashMap<String, AboutMem>, Comps: &Vec<Comp>, Regs: &mut HashMap<String, AboutReg>, Adders: &mut HashMap<String, AboutAdder>, LTs: &mut HashMap<String, AboutLT>) {
    tap(output_file);

    let cells = String::from("cells {\n");

    output_file.push_str(&cells);
    
    for (name, about) in Mems {
        let memory = match about {
            AboutMem::Mem1D{ .. } => {
                String::from("@external(1) ")
                + &name
                + " = std_mem_d"
                + &String::from("1")
                + "("
                + &(&about.width()).to_string()
                + ", "
                + &(&about.size()).to_string()
                + ", "
                + &(&about.idx_size()).to_string()
                + ");\n"
            },
            AboutMem::Mem2D{ .. } => {
                String::from("@external(1) ")
                + &name
                + " = std_mem_d"
                + &String::from("2")
                + "("
                + &(&about.width()).to_string()
                + ", "
                + &(&about.dim0_size()).to_string()
                + ", "
                + &(&about.dim1_size()).to_string()
                + ", "
                + &(&about.idx_size()).to_string()
                + ", "
                + &(&about.idx_size()).to_string()
                + ");\n"
            },
            _ => unreachable!()
        };

        tap(output_file); tap(output_file);
        output_file.push_str(&memory);
    }

    println!("\n\nPrinting Comps:\n");

    for comp in Comps {
        println!("{:?}", &comp);
        match comp.op {
            Op::mult => {
                let accumulator =
                    match ID_HashMap.get(&comp.right).unwrap().vec.vec_type() {  // vec_type 확인해서 int면 std_add, float면 std_fp_add
                        VecType::int => String::from("MAC_accumulator = std_add("),
                        VecType::float => String::from("MAC_accumulator = std_fp_add("),
                        _ => unreachable!()
                    }
                    + &String::from("32")
                    + &String::from(");\n");
                
                tap(output_file); tap(output_file);
                output_file.push_str(&accumulator);
                
                let multiplier = 
                    match ID_HashMap.get(&comp.right).unwrap().vec.vec_type() {
                        VecType::int => String::from("MAC_multiplier = std_mult_pipe("),
                        VecType::float => String::from("MAC_multiplier = std_fp_mult_pipe("),
                        _ => unreachable!()
                    }
                    + &String::from("32")
                    + &String::from(");\n");
                
                tap(output_file); tap(output_file);
                output_file.push_str(&multiplier);

                let name_W_idx0 = comp.left.clone() + &String::from("_idx0");
                let W_idx0 = AboutReg {
                    width: 32,

                    in_: name_W_idx0.clone() + &String::from(".in"),
                    write_en: name_W_idx0.clone() + &String::from(".write_en"),
                    out: name_W_idx0.clone() + &String::from(".out"),
                    done: name_W_idx0.clone() + &String::from(".done")
                };

                let name_W_idx1 = comp.left.clone() + &String::from("_idx1");
                let W_idx1 = AboutReg {
                    width: 32,

                    in_: name_W_idx1.clone() + &String::from(".in"),
                    write_en: name_W_idx1.clone() + &String::from(".write_en"),
                    out: name_W_idx1.clone() + &String::from(".out"),
                    done: name_W_idx1.clone() + &String::from(".done")
                };

                let name_x_idx1 = comp.right.clone() + &String::from("_idx1");
                let x_idx1 = AboutReg {
                    width: 32,

                    in_: name_x_idx1.clone() + &String::from(".in"),
                    write_en: name_x_idx1.clone() + &String::from(".write_en"),
                    out: name_x_idx1.clone() + &String::from(".out"),
                    done: name_x_idx1.clone() + &String::from(".done")
                };

                let name_M_idx = comp.out.clone() + &String::from("_idx");
                let m_idx = AboutReg {
                    width: 32,

                    in_: name_M_idx.clone() + &String::from(".in"),
                    write_en: name_M_idx.clone() + &String::from(".write_en"),
                    out: name_M_idx.clone() + &String::from(".out"),
                    done: name_M_idx.clone() + &String::from(".done")
                };

                Regs.entry(name_W_idx0).or_insert(W_idx0);
                Regs.entry(name_W_idx1).or_insert(W_idx1);
                Regs.entry(name_x_idx1).or_insert(x_idx1);
                Regs.entry(name_M_idx).or_insert(m_idx);

                let name_W_idx0_adder = comp.left.clone() + &String::from("_idx0_adder");
                let W_idx0_adder = AboutAdder {
                    width: 32,

                    left: name_W_idx0_adder.clone() + &String::from(".left"),
                    right: name_W_idx0_adder.clone() + &String::from(".right"),
                    out: name_W_idx0_adder.clone() + &String::from(".out")
                };

                let name_W_idx1_adder = comp.left.clone() + &String::from("_idx1_adder");
                let W_idx1_adder = AboutAdder {
                    width: 32,

                    left: name_W_idx1_adder.clone() + &String::from(".left"),
                    right: name_W_idx1_adder.clone() + &String::from(".right"),
                    out: name_W_idx1_adder.clone() + &String::from(".out")
                };

                let name_x_idx1_adder = comp.right.clone() + &String::from("_idx1_adder");
                let x_idx1_adder = AboutAdder {
                    width: 32,

                    left: name_x_idx1_adder.clone() + &String::from(".left"),
                    right: name_x_idx1_adder.clone() + &String::from(".right"),
                    out: name_x_idx1_adder.clone() + &String::from(".out")
                };

                let name_M_idx_adder = comp.out.clone() + &String::from("_idx_adder");
                let m_idx_adder = AboutAdder {
                    width: 32,

                    left: name_M_idx_adder.clone() + &String::from(".left"),
                    right: name_M_idx_adder.clone() + &String::from(".right"),
                    out: name_M_idx_adder.clone() + &String::from(".out")
                };
                
                Adders.entry(name_W_idx0_adder).or_insert(W_idx0_adder);
                Adders.entry(name_W_idx1_adder).or_insert(W_idx1_adder);
                Adders.entry(name_x_idx1_adder).or_insert(x_idx1_adder);
                Adders.entry(name_M_idx_adder).or_insert(m_idx_adder);

                let name_W_idx0_lt = comp.left.clone() + &String::from("_idx0_lt");
                let W_idx0_lt = AboutLT {
                    width: 32,

                    left: name_W_idx0_lt.clone() + &String::from(".left"),
                    right: name_W_idx0_lt.clone() + &String::from(".right"),
                    out: name_W_idx0_lt.clone() + &String::from(".out")
                };

                let name_W_idx1_lt = comp.left.clone() + &String::from("_idx1_lt");
                let W_idx1_lt = AboutLT {
                    width: 32,

                    left: name_W_idx1_lt.clone() + &String::from(".left"),
                    right: name_W_idx1_lt.clone() + &String::from(".right"),
                    out: name_W_idx1_lt.clone() + &String::from(".out")
                };

                let name_x_idx1_lt = comp.right.clone() + &String::from("_idx1_lt");
                let x_idx1_lt = AboutLT {
                    width: 32,

                    left: name_x_idx1_lt.clone() + &String::from(".left"),
                    right: name_x_idx1_lt.clone() + &String::from(".right"),
                    out: name_x_idx1_lt.clone() + &String::from(".out")
                };

                let name_M_idx_lt = comp.out.clone() + &String::from("_idx_lt");
                let m_idx_lt = AboutLT {
                    width: 32,

                    left: name_M_idx_lt.clone() + &String::from(".left"),
                    right: name_M_idx_lt.clone() + &String::from(".right"),
                    out: name_M_idx_lt.clone() + &String::from(".out")
                };

                LTs.entry(name_W_idx0_lt).or_insert(W_idx0_lt);
                LTs.entry(name_W_idx1_lt).or_insert(W_idx1_lt);
                LTs.entry(name_x_idx1_lt).or_insert(x_idx1_lt);
                LTs.entry(name_M_idx_lt).or_insert(m_idx_lt);
            },
            Op::add => {
                let adder = 
                    match ID_HashMap.get(&comp.right).unwrap().vec.vec_type() {
                        VecType::int => String::from("MAC_adder = std_add("),
                        VecType::float => String::from("MAC_adder = std_fp_add("),
                        _ => unreachable!()
                    }
                    + &String::from("32")
                    + &String::from(");\n");
                
                tap(output_file); tap(output_file);
                output_file.push_str(&adder);

                let name_b_idx0 = comp.right.clone() + &String::from("_idx0");
                let b_idx0 = AboutReg {
                    width: 32,

                    in_: name_b_idx0.clone() + &String::from(".in"),
                    write_en: name_b_idx0.clone() + &String::from(".write_en"),
                    out: name_b_idx0.clone() + &String::from(".out"),
                    done: name_b_idx0.clone() + &String::from(".done")
                };

                let name_b_idx1 = comp.right.clone() + &String::from("_idx1");
                let b_idx1 = AboutReg {
                    width: 32,

                    in_: name_b_idx1.clone() + &String::from(".in"),
                    write_en: name_b_idx1.clone() + &String::from(".write_en"),
                    out: name_b_idx1.clone() + &String::from(".out"),
                    done: name_b_idx1.clone() + &String::from(".done")
                };

                Regs.entry(name_b_idx0).or_insert(b_idx0);
                Regs.entry(name_b_idx1).or_insert(b_idx1);
            },
            _ => unreachable!()
        };
    }

    for (name, about) in Regs {
        let reg =
            name.clone()
            + &String::from(" = std_reg(") 
            + &about.width.to_string()
            + &String::from(");\n");

        tap(output_file); tap(output_file);
        output_file.push_str(&reg);
    }

    for (name, about) in Adders {
        let adder =
            name.clone()
            + &String::from(" = std_add(") 
            + &about.width.to_string()
            + &String::from(");\n");

        tap(output_file); tap(output_file);
        output_file.push_str(&adder);
    }

    for (name, about) in LTs {
        let lt =
            name.clone()
            + &String::from(" = std_lt(") 
            + &about.width.to_string()
            + &String::from(");\n");

        tap(output_file); tap(output_file);
        output_file.push_str(&lt);
    }

    tap(output_file);
    output_file.push_str("}\n");
}

fn generate_Wires(output_file: &mut String, ASTNodes: &Vec<ASTNode>, ID_HashMap: &HashMap<String, AboutID>,
                  Mems: &HashMap<String, AboutMem>, Comps: &Vec<Comp>, Regs: &HashMap<String, AboutReg>, Adders: &HashMap<String, AboutAdder>, LTs: &HashMap<String, AboutLT>) {
    
    tap(output_file);

    let wires = String::from("wires {\n");

    output_file.push_str(&wires);

    generate_Group_cond(output_file, ID_HashMap, Comps, Regs, LTs);
    generate_Group_incr(output_file, Comps, Regs, Adders);
    generate_Group_init(output_file, Comps, Regs);
    
    for node in ASTNodes {  // Wire ASTNode마다, 즉 MAC() 하나마다 product, accumulate, plus가 하나씩 필요
        match node {
            ASTNode::Decl { .. } => (),
            ASTNode::Wire { .. } => {
                let output = node.name();

                let mac = node.MAC();

                let weight = mac.weight();
                let input = mac.input();
                let bias = mac.bias();

                let M = String::from("M") + "_" + &weight + "_" + &input;

                generate_Group_product(output_file, Mems, Regs, &weight, &input, &M);
                generate_Group_accumulate(output_file, Mems, Regs, &weight, &input, &M, &output);
                generate_Group_plus(output_file, Mems, Regs, &weight, &input, &bias, &output);
            },
            _ => unreachable!()
        };
    }

    tap(output_file);
    output_file.push_str("}\n");
}

fn connect(output_file: &mut String, connections: &Vec<(String, String)>) {
    for connection in connections {
        tap(output_file); tap(output_file); tap(output_file);
        output_file.push_str(&(connection.0.clone() + " = " + &connection.1));
        output_file.push_str(";\n");
    }
}

fn generate_Group_cond(output_file: &mut String, ID_HashMap: &HashMap<String, AboutID>, Comps: &Vec<Comp>, Regs: &HashMap<String, AboutReg>, LTs: &HashMap<String, AboutLT>) {
    fn generate_Cond(output_file: &mut String, ID_HashMap: &HashMap<String, AboutID>, Regs: &HashMap<String, AboutReg>, LTs: &HashMap<String, AboutLT>, comp: &Comp, idx: &String, ID: &String) {
        let cond_name = String::from("cond_") + &idx;
        let group_cond = String::from("group ") + &cond_name + &String::from(" {\n");

        tap(output_file); tap(output_file);
        output_file.push_str(&group_cond);
        
        let lt = idx.clone() + "_lt";

        let connections = Vec::from([
            (LTs.get(&lt).unwrap().left.clone(), Regs.get(idx).unwrap().out.clone()),
            match idx.chars().last().unwrap() {  // W_idx0_lt를 key로 하여 HashMap을 타고 들어가 lt.left를 connect에 넣는다.
                '0' => (LTs.get(&lt).unwrap().right.clone(), (String::from("32'd") + &ID_HashMap.get(ID).unwrap().vec.dim0().to_string()).clone()),
                '1' => (LTs.get(&lt).unwrap().right.clone(), (String::from("32'd") + &ID_HashMap.get(ID).unwrap().vec.dim1().to_string()).clone()),
                'x' => (LTs.get(&lt).unwrap().right.clone(), (String::from("32'd") + &ID_HashMap.get(&comp.left).unwrap().vec.dim1().to_string()).clone()),
                _ => unreachable!()
            },
            ((cond_name + &String::from("[done]")), String::from("1'd1"))
        ]);

        connect(output_file, &connections);

        tap(output_file); tap(output_file);
        output_file.push_str("}\n");
    }

    for comp in Comps {  // call generate_Cond(), MAC operation에서는 mult에서만 group cond가 4개 필요(W_idx0, W_idx1, x_idx1, m_idx)
        match comp.op {
            Op::mult => {
                let mut idx: String;
                let idx = comp.left.clone() + "_idx0";
                generate_Cond(output_file, ID_HashMap, Regs, LTs, &comp, &idx, &comp.left);
                let idx = comp.left.clone() + "_idx1";
                generate_Cond(output_file, ID_HashMap, Regs, LTs, &comp, &idx, &comp.left);
                let idx = comp.right.clone() + "_idx1";
                generate_Cond(output_file, ID_HashMap, Regs, LTs, &comp, &idx, &comp.right);
                let idx = comp.out.clone() + "_idx";
                generate_Cond(output_file, ID_HashMap, Regs, LTs, &comp, &idx, &comp.out);
            },
            Op::add => (),
            _ => unreachable!()
        };
    }
}

fn generate_Group_incr(output_file: &mut String, Comps: &Vec<Comp>, Regs: &HashMap<String, AboutReg>, Adders: &HashMap<String, AboutAdder>) {
    fn generate_Incr(output_file: &mut String, Regs: &HashMap<String, AboutReg>, Adders: &HashMap<String, AboutAdder>, idx: &String) {
        let incr_name = String::from("incr_") + &idx;
        let group_incr = String::from("group ") + &incr_name + &String::from(" {\n");

        tap(output_file); tap(output_file);
        output_file.push_str(&group_incr);
        
        let adder = idx.clone() + "_adder";
        
        let connections = Vec::from([
            (Adders.get(&adder).unwrap().left.clone(), Regs.get(idx).unwrap().out.clone()),
            (Adders.get(&adder).unwrap().right.clone(), String::from("32'd1")),
            (Regs.get(idx).unwrap().write_en.clone(), String::from("1'd1")),
            (Regs.get(idx).unwrap().in_.clone(), Adders.get(&adder).unwrap().out.clone()),
            ((incr_name + &String::from("[done]")), Regs.get(idx).unwrap().done.clone())
        ]);

        connect(output_file, &connections);

        tap(output_file); tap(output_file);
        output_file.push_str("}\n");
    }

    for comp in Comps {  // call generate_Incr(), group incr는 cond마다 하나씩 4개 필요
        match comp.op {
            Op::mult => {
                let mut idx: String;
                let idx = comp.left.clone() + "_idx0";
                generate_Incr(output_file, Regs, Adders, &idx);
                let idx = comp.left.clone() + "_idx1";
                generate_Incr(output_file, Regs, Adders, &idx);
                let idx = comp.right.clone() + "_idx1";
                generate_Incr(output_file, Regs, Adders, &idx);
                let idx = comp.out.clone() + "_idx";
                generate_Incr(output_file, Regs, Adders, &idx);
            },
            Op::add => (),
            _ => unreachable!()
        };
    }
}

fn generate_Group_init(output_file: &mut String, Comps: &Vec<Comp>, Regs: &HashMap<String, AboutReg>) {
    fn generate_Init(output_file: &mut String, Regs: &HashMap<String, AboutReg>, idx: &String) {
        let init_name = String::from("init_") + &idx;
        let group_init = String::from("group ") + &init_name + &String::from(" {\n");

        tap(output_file); tap(output_file);
        output_file.push_str(&group_init);
        
        let connections = Vec::from([
            (Regs.get(idx).unwrap().write_en.clone(), String::from("1'd1")),
            (Regs.get(idx).unwrap().in_.clone(), String::from("32'd0")),
            ((init_name + &String::from("[done]")), Regs.get(idx).unwrap().done.clone())
        ]);

        connect(output_file, &connections);

        tap(output_file); tap(output_file);
        output_file.push_str("}\n");
    }

    for comp in Comps {  // call generate_Init(),  group init은 cond마다 하나씩 4개 필요
        match comp.op {
            Op::mult => {
                let mut idx: String;
                let idx = comp.left.clone() + "_idx0";
                generate_Init(output_file, Regs, &idx);
                let idx = comp.left.clone() + "_idx1";
                generate_Init(output_file, Regs, &idx);
                let idx = comp.right.clone() + "_idx1";
                generate_Init(output_file, Regs, &idx);
                let idx = comp.out.clone() + "_idx";
                generate_Init(output_file, Regs, &idx);
            },
            Op::add => (),
            _ => unreachable!()
        };
    }
}

fn generate_Group_product(output_file: &mut String, Mems: &HashMap<String, AboutMem>, Regs: &HashMap<String, AboutReg>, weight: &String, input: &String, M: &String) {
    let product_name = String::from("product_") + &weight + "_" + &input;
    let group_product = String::from("group ") + &product_name + &String::from(" {\n");

    tap(output_file); tap(output_file);
    output_file.push_str(&group_product);

    let connections = Vec::from([
        (Mems.get(M).unwrap().write_en().clone(), String::from("MAC_multiplier.done")),
        (String::from("MAC_multiplier.left"), Mems.get(weight).unwrap().read_data().clone()),
        (String::from("MAC_multiplier.right"), Mems.get(input).unwrap().read_data().clone()),
        (String::from("MAC_multiplier.go"), String::from("!MAC_multiplier.done ? 1'd1")),
        (Mems.get(M).unwrap().write_data().clone(), String::from("MAC_multiplier.out")),
        ((product_name + &String::from("[done]")), Mems.get(M).unwrap().done().clone()),
        (Mems.get(weight).unwrap().addr0().clone(), Regs.get(&(weight.clone() + "_idx0")).unwrap().out.clone()),
        (Mems.get(weight).unwrap().addr1().clone(), Regs.get(&(weight.clone() + "_idx1")).unwrap().out.clone()),
        (Mems.get(input).unwrap().addr0().clone(), Regs.get(&(weight.clone() + "_idx1")).unwrap().out.clone()),
        (Mems.get(input).unwrap().addr1().clone(), Regs.get(&(input.clone() + "_idx1")).unwrap().out.clone()),
        (Mems.get(M).unwrap().addr0().clone(), Regs.get(&(weight.clone() + "_idx1")).unwrap().out.clone())
    ]);

    connect(output_file, &connections);

    tap(output_file); tap(output_file);
    output_file.push_str("}\n");
}

fn generate_Group_accumulate(output_file: &mut String, Mems: &HashMap<String, AboutMem>, Regs: &HashMap<String, AboutReg>, weight: &String, input: &String, M: &String, output: &String) {
    let accumulate_name = String::from("accumulate_") + &M;
    let group_accumulate = String::from("group ") + &accumulate_name + &String::from(" {\n");

    tap(output_file); tap(output_file);
    output_file.push_str(&group_accumulate);

    let connections = Vec::from([
        (Mems.get(output).unwrap().write_en().clone(), String::from("1'd1")),
        (String::from("MAC_accumulator.left"), Mems.get(output).unwrap().read_data().clone()),
        (String::from("MAC_accumulator.right"), Mems.get(M).unwrap().read_data().clone()),
        (Mems.get(output).unwrap().write_data().clone(), String::from("MAC_accumulator.out")),
        ((accumulate_name + &String::from("[done]")), Mems.get(output).unwrap().done().clone()),
        (Mems.get(M).unwrap().addr0().clone(), Regs.get(&(M.clone() + "_idx")).unwrap().out.clone()),
        (Mems.get(output).unwrap().addr0().clone(), Regs.get(&(weight.clone() + "_idx0")).unwrap().out.clone()),
        (Mems.get(output).unwrap().addr1().clone(), Regs.get(&(input.clone() + "_idx1")).unwrap().out.clone())
    ]);

    connect(output_file, &connections);

    tap(output_file); tap(output_file);
    output_file.push_str("}\n");
}

fn generate_Group_plus(output_file: &mut String, Mems: &HashMap<String, AboutMem>, Regs: &HashMap<String, AboutReg>, weight: &String, input: &String, bias: &String, output: &String) {
    let plus_name = String::from("plus_") + &output + "_" + &bias;
    let group_plus = String::from("group ") + &plus_name + &String::from(" {\n");

    tap(output_file); tap(output_file);
    output_file.push_str(&group_plus);

    let connections = Vec::from([
        (Mems.get(output).unwrap().write_en().clone(), String::from("1'd1")),
        (String::from("MAC_adder.left"), Mems.get(output).unwrap().read_data().clone()),
        (String::from("MAC_adder.right"), Mems.get(bias).unwrap().read_data().clone()),
        (Mems.get(output).unwrap().write_data().clone(), String::from("MAC_adder.out")),
        ((plus_name + &String::from("[done]")), Mems.get(output).unwrap().done().clone()),
        (Mems.get(bias).unwrap().addr0().clone(), Regs.get(&(weight.clone() + "_idx0")).unwrap().out.clone()),
        (Mems.get(bias).unwrap().addr1().clone(), Regs.get(&(input.clone() + "_idx1")).unwrap().out.clone()),
        (Mems.get(output).unwrap().addr0().clone(), Regs.get(&(weight.clone() + "_idx0")).unwrap().out.clone()),
        (Mems.get(output).unwrap().addr1().clone(), Regs.get(&(input.clone() + "_idx1")).unwrap().out.clone())
    ]);

    connect(output_file, &connections);

    tap(output_file); tap(output_file);
    output_file.push_str("}\n");
}

pub enum OperType {
    iter,
    mac
}

pub struct AboutOper {
    oper_type: OperType,
    nested_opers: Vec<String>
}

fn generate_Control(output_file: &mut String, ASTNodes: &Vec<ASTNode>) {
    let control = String::from("control {\n");
    let seq = String::from("seq {\n");

    tap(output_file);
    output_file.push_str(&control);
    tap(output_file); tap(output_file);
    output_file.push_str(&seq);

    for node in ASTNodes {
        match node {
            ASTNode::Decl { .. } => (),
            ASTNode::Wire { .. } => {
                let output = node.name();

                let mac = node.MAC();

                let weight = mac.weight();
                let input = mac.input();
                let bias = mac.bias();

                let M = String::from("M") + "_" + &weight + "_" + &input;

                let row_idx = weight.clone() + &String::from("_idx0");
                let column_idx = input.clone() + &String::from("_idx1");
                let product_idx = weight.clone() + &String::from("_idx1");
                let accumulate_idx = M.clone() + &String::from("_idx");
                let product = String::from("product_") + &weight + "_" + &input;
                let accumulate = String::from("accumulate_") + &M;
                let plus = String::from("plus_") + &output + "_" + &bias;

                let mut Opers: HashMap<String, AboutOper> = HashMap::new();

                Opers.entry(row_idx.clone()).or_insert(AboutOper { oper_type: OperType::iter, nested_opers: Vec::from([column_idx.clone()]) });
                Opers.entry(column_idx).or_insert(AboutOper { oper_type: OperType::iter, nested_opers: Vec::from([product_idx.clone(), accumulate_idx.clone(), plus.clone()]) });
                Opers.entry(product_idx).or_insert(AboutOper { oper_type: OperType::iter, nested_opers: Vec::from([product.clone()]) });
                Opers.entry(accumulate_idx).or_insert(AboutOper { oper_type: OperType::iter, nested_opers: Vec::from([accumulate.clone()]) });
                Opers.entry(product).or_insert(AboutOper { oper_type: OperType::mac, nested_opers: Vec::new() });
                Opers.entry(accumulate).or_insert(AboutOper { oper_type: OperType::mac, nested_opers: Vec::new() });
                Opers.entry(plus).or_insert(AboutOper { oper_type: OperType::mac, nested_opers: Vec::new() });

                generate_Oper(output_file, &Opers, &row_idx, 3);
            },
            _ => unreachable!()
        };
    }

    tap(output_file); tap(output_file);
    output_file.push_str("}\n");
    tap(output_file);
    output_file.push_str("}\n");
}

fn generate_Iter(output_file: &mut String, Opers: &HashMap<String, AboutOper>, idx: &String, nested_opers: &Vec<String>, nest: i8) {
    let init = String::from("init_") + &idx;
    let lt = idx.clone() + &String::from("_lt.out");
    let cond = String::from("cond_") + &idx;
    let while_ = String::from("while ") + &lt + &String::from(" with ") + &cond + &String::from(" {\n");
    let seq = String::from("seq {\n");
    let incr = String::from("incr_") + &idx;

    for i in 0..nest { tap(output_file); }
    output_file.push_str(&init);
    output_file.push_str(";\n");

    for i in 0..nest { tap(output_file); }
    output_file.push_str(&while_);

    for i in 0..nest+1 { tap(output_file); }
    output_file.push_str(&seq);

    for oper in nested_opers {
        generate_Oper(output_file, Opers, oper, nest+2);
    }
    
    for i in 0..nest+2 { tap(output_file); }
    output_file.push_str(&incr);
    output_file.push_str(";\n");
    
    for i in 0..nest+1 { tap(output_file); }
    output_file.push_str("}\n");

    for i in 0..nest { tap(output_file); }
    output_file.push_str("}\n");
}

fn generate_Oper(output_file: &mut String, Opers: &HashMap<String, AboutOper>, oper: &String, nest: i8) {
    match Opers.get(oper).unwrap().oper_type {
        OperType::iter => {
            let nested_opers = &Opers.get(oper).unwrap().nested_opers;
            generate_Iter(output_file, Opers, oper, &nested_opers, nest);  // iter operation은 internal node -> nested call
        },
        OperType::mac => {
            for i in 0..nest { tap(output_file); }
            output_file.push_str(&oper);                                   // mac operation은 leaf node -> no nested call
            output_file.push_str(";\n");
        },
        _ => unreachable!()
    };
}

//======================================== Calyx Generator ========================================//


fn main() {  // Parser를 통해 만든 AST(Vector<ASTNode>)를 iter하며 Mems, Comps, Regs, Adders, LTs 완성 -> Calyx Generator를 통해 Compile
    let input_file = std::fs::read_to_string("input.j21").expect("Cannot read file!");
    
    let mut output_file = String::new();

    let mut ID_HashMap: HashMap<String, AboutID> = HashMap::new();

    let ASTNodes = parse(&input_file, &mut ID_HashMap);  // parse()에서 J21Parser를 가지고 parsing

    let mut Mems: HashMap<String, AboutMem> = HashMap::new();
    let mut Comps: Vec<Comp> = Vec::new();

    let mut Regs: HashMap<String, AboutReg> = HashMap::new();
    let mut Adders: HashMap<String, AboutAdder> = HashMap::new();
    let mut LTs: HashMap<String, AboutLT> = HashMap::new();

    // Tests for printing Nodes
    println!("\n\nTests for printing ASTNode:\n");
    println!("{:?}", &ASTNodes[3]);
    println!("{:?}", &ASTNodes[3].port_type());
    println!("{:?}", &ASTNodes[3].name());
    println!("{:?}", &ASTNodes[3].MAC());
    println!("{:?}", &ASTNodes[3].MAC().weight());
    println!("{:?}", &ASTNodes[3].MAC().input());
    println!("{:?}", &ASTNodes[3].MAC().bias());
    println!("{:?}", ID_HashMap.get(&ASTNodes[3].MAC().weight()));
    println!("{:?}", ID_HashMap.get(&ASTNodes[3].MAC().weight()).unwrap().vec.dim0());
    
    println!("\n\nPrinting all nodes:\n");

    for node in &ASTNodes {  // AST를 iter해 collect_Mem(), collect_Comp()
        println!("{:?}", &node);
        match node {
            ASTNode::Decl { .. } => {
                let name = node.name();

                let vec = node.Vector();

                collect_Mem(&mut Mems, &name, 2, None, &Some(vec));
            },
            ASTNode::Wire { .. } => {
                let name_y = node.name();

                let mac = node.MAC();

                let name_M = String::from("M") + "_" + &mac.weight() + "_" + &mac.input();

                let size_m = ID_HashMap.get(&mac.weight()).unwrap().vec.dim1();

                let vec_y = ASTNode::Vector {
                    vec_type: ID_HashMap.get(&mac.bias()).unwrap().vec.vec_type(),
                    dim0: ID_HashMap.get(&mac.weight()).unwrap().vec.dim0(),
                    dim1: ID_HashMap.get(&mac.input()).unwrap().vec.dim1()
                };

                collect_Mem(&mut Mems, &name_M, 1, Some(size_m), &None);
                collect_Mem(&mut Mems, &name_y, 2, None, &Some(vec_y));
                collect_Comp(&mut Comps, &name_M, &name_y, &mac);
            },
            _ => unreachable!()
        };
    }
    
    Calyx_Generator(&mut output_file, &ASTNodes, &ID_HashMap, &Mems, &Comps, &mut Regs, &mut Adders, &mut LTs);
    println!("\n\nGenerated Calyx:\n\n{}", output_file);
    
    std::fs::write("j21.futil", output_file).expect("Unable to write file");
}