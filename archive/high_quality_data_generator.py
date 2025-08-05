#!/usr/bin/env python3
"""
High-Quality Diverse Data Generator for WordPiece Training

This script generates much more diverse training data by:
1. Using varied vocabulary from multiple domains
2. Creating natural language variations
3. Incorporating technical terms, proper nouns, and edge cases
4. Ensuring high lexical diversity for effective BPE training
"""

import random
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HighQualityDataGenerator:
    """Generate high-diversity training data for tokenizer training."""
    
    def __init__(self):
        # Base vocabulary pools from different domains
        self.technical_terms = [
            "algorithm", "optimization", "implementation", "architecture", "framework",
            "methodology", "paradigm", "infrastructure", "scalability", "performance",
            "authentication", "authorization", "encryption", "decryption", "cryptography",
            "virtualization", "containerization", "orchestration", "deployment", "monitoring",
            "microservices", "serverless", "distributed", "parallel", "concurrent",
            "asynchronous", "synchronous", "multithreading", "multiprocessing", "pipeline"
        ]
        
        self.scientific_terms = [
            "hypothesis", "methodology", "experiment", "analysis", "synthesis",
            "investigation", "observation", "measurement", "calibration", "validation",
            "verification", "documentation", "publication", "peer-review", "reproducibility",
            "statistical", "empirical", "theoretical", "computational", "mathematical",
            "biological", "chemical", "physical", "astronomical", "geological",
            "environmental", "ecological", "evolutionary", "genetic", "molecular"
        ]
        
        self.business_terms = [
            "strategy", "execution", "implementation", "governance", "compliance",
            "stakeholder", "shareholder", "customer", "client", "vendor",
            "partnership", "collaboration", "negotiation", "acquisition", "merger",
            "revenue", "profitability", "sustainability", "scalability", "efficiency",
            "productivity", "innovation", "transformation", "digitalization", "automation",
            "analytics", "intelligence", "insights", "forecasting", "budgeting"
        ]
        
        self.academic_terms = [
            "dissertation", "thesis", "research", "scholarship", "fellowship",
            "curriculum", "pedagogy", "assessment", "evaluation", "accreditation",
            "interdisciplinary", "multidisciplinary", "undergraduate", "graduate", "postgraduate",
            "professor", "lecturer", "instructor", "administrator", "registrar",
            "enrollment", "admission", "graduation", "certification", "qualification",
            "seminar", "symposium", "conference", "workshop", "colloquium"
        ]
        
        self.medical_terms = [
            "diagnosis", "treatment", "therapy", "medication", "prescription",
            "symptom", "syndrome", "pathology", "physiology", "anatomy",
            "cardiology", "neurology", "oncology", "pediatrics", "geriatrics",
            "immunology", "endocrinology", "dermatology", "psychiatry", "radiology",
            "surgery", "anesthesia", "rehabilitation", "prevention", "screening",
            "vaccination", "immunization", "antibiotic", "antiviral", "chemotherapy"
        ]
        
        self.proper_nouns = [
            # Companies
            "Microsoft", "Google", "Amazon", "Apple", "Tesla", "Netflix", "Meta",
            "OpenAI", "Anthropic", "DeepMind", "NVIDIA", "Intel", "AMD", "IBM",
            # Universities
            "Harvard", "MIT", "Stanford", "Berkeley", "Princeton", "Yale", "Oxford",
            "Cambridge", "Caltech", "Carnegie", "Cornell", "Columbia", "Chicago",
            # Cities
            "NewYork", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto",
            "Singapore", "Beijing", "Mumbai", "Barcelona", "Amsterdam", "Stockholm",
            # Countries
            "UnitedStates", "UnitedKingdom", "Germany", "France", "Japan", "China",
            "Australia", "Canada", "Netherlands", "Sweden", "Switzerland", "Denmark"
        ]
        
        self.connecting_words = [
            "and", "or", "but", "however", "therefore", "moreover", "furthermore",
            "nevertheless", "consequently", "additionally", "specifically", "particularly",
            "especially", "including", "such as", "for example", "in particular",
            "as well as", "in addition to", "combined with", "together with",
            "along with", "coupled with", "integrated with", "associated with"
        ]
        
        self.sentence_starters = [
            "Advanced", "Innovative", "Comprehensive", "Sophisticated", "Revolutionary",
            "Cutting-edge", "State-of-the-art", "Next-generation", "Breakthrough",
            "Pioneering", "Groundbreaking", "Transformative", "Disruptive", "Emerging",
            "Contemporary", "Modern", "Progressive", "Dynamic", "Adaptive", "Flexible",
            "Scalable", "Robust", "Reliable", "Efficient", "Effective", "Optimal",
            "Strategic", "Tactical", "Systematic", "Methodical", "Analytical", "Critical"
        ]
        
        self.technical_adjectives = [
            "distributed", "centralized", "decentralized", "federated", "hierarchical",
            "modular", "monolithic", "microservice-based", "cloud-native", "hybrid",
            "multi-tenant", "single-tenant", "cross-platform", "platform-agnostic",
            "vendor-neutral", "open-source", "proprietary", "enterprise-grade",
            "mission-critical", "fault-tolerant", "high-availability", "low-latency",
            "real-time", "near-real-time", "batch-processed", "stream-processed"
        ]
        
        # Suffixes and prefixes for word variation
        self.prefixes = ["pre", "post", "anti", "pro", "inter", "intra", "trans", "multi", "uni", "bi", "tri"]
        self.suffixes = ["tion", "sion", "ment", "ness", "ity", "ism", "ist", "er", "or", "ive", "able", "ible"]
        
    def generate_technical_sentence(self):
        """Generate a technical sentence with varied vocabulary."""
        starter = random.choice(self.sentence_starters)
        adj = random.choice(self.technical_adjectives)
        tech_term = random.choice(self.technical_terms)
        connector = random.choice(self.connecting_words)
        second_term = random.choice(self.technical_terms + self.scientific_terms)
        
        # Create variations
        variations = [
            f"{starter} {adj} {tech_term} systems enable {connector} {second_term} capabilities",
            f"{starter} {tech_term} {connector} {second_term} require {adj} implementation strategies",
            f"The {adj} {tech_term} approach facilitates {second_term} optimization {connector} performance enhancement",
            f"Modern {tech_term} architectures incorporate {adj} {second_term} methodologies",
            f"{starter} organizations leverage {adj} {tech_term} solutions for {second_term} advancement"
        ]
        
        return random.choice(variations)
    
    def generate_scientific_sentence(self):
        """Generate a scientific sentence with domain-specific vocabulary."""
        sci_term1 = random.choice(self.scientific_terms)
        sci_term2 = random.choice(self.scientific_terms)
        connector = random.choice(self.connecting_words)
        adj = random.choice(self.technical_adjectives)
        
        variations = [
            f"Recent {sci_term1} studies demonstrate {adj} correlations between {sci_term2} variables",
            f"The {adj} {sci_term1} methodology enables comprehensive {sci_term2} analysis",
            f"Researchers conducting {sci_term1} investigations utilize {adj} {sci_term2} techniques",
            f"Contemporary {sci_term1} approaches integrate {sci_term2} frameworks {connector} analytical methods",
            f"Advanced {sci_term1} protocols facilitate {adj} {sci_term2} measurements {connector} data collection"
        ]
        
        return random.choice(variations)
    
    def generate_business_sentence(self):
        """Generate a business sentence with corporate vocabulary."""
        biz_term1 = random.choice(self.business_terms)
        biz_term2 = random.choice(self.business_terms)
        proper_noun = random.choice(self.proper_nouns)
        adj = random.choice(self.technical_adjectives)
        
        variations = [
            f"Enterprise {biz_term1} initiatives require {adj} {biz_term2} frameworks for successful implementation",
            f"{proper_noun} demonstrates {adj} {biz_term1} excellence through innovative {biz_term2} solutions",
            f"Global organizations prioritize {biz_term1} optimization {random.choice(self.connecting_words)} {biz_term2} enhancement",
            f"Strategic {biz_term1} planning incorporates {adj} {biz_term2} methodologies for competitive advantage",
            f"Leading companies like {proper_noun} invest in {adj} {biz_term1} capabilities {random.choice(self.connecting_words)} {biz_term2} transformation"
        ]
        
        return random.choice(variations)
    
    def generate_academic_sentence(self):
        """Generate an academic sentence with educational vocabulary."""
        acad_term1 = random.choice(self.academic_terms)
        acad_term2 = random.choice(self.academic_terms)
        proper_noun = random.choice([p for p in self.proper_nouns if any(u in p for u in ["Harvard", "MIT", "Stanford", "Berkeley", "Princeton", "Yale", "Oxford", "Cambridge"])])
        
        variations = [
            f"Distinguished {acad_term1} programs at {proper_noun} emphasize {acad_term2} excellence through rigorous curriculum",
            f"International {acad_term1} collaboration facilitates cross-cultural {acad_term2} exchange programs",
            f"Academic {acad_term1} standards require comprehensive {acad_term2} assessment methodologies",
            f"Premier institutions like {proper_noun} offer specialized {acad_term1} opportunities in {acad_term2}",
            f"Contemporary {acad_term1} approaches integrate technology-enhanced {acad_term2} delivery methods"
        ]
        
        return random.choice(variations)
    
    def generate_medical_sentence(self):
        """Generate a medical sentence with healthcare vocabulary."""
        med_term1 = random.choice(self.medical_terms)
        med_term2 = random.choice(self.medical_terms)
        adj = random.choice(self.technical_adjectives)
        
        variations = [
            f"Advanced {med_term1} protocols enhance patient {med_term2} outcomes through evidence-based approaches",
            f"Clinical {med_term1} research demonstrates {adj} improvements in {med_term2} effectiveness",
            f"Healthcare professionals utilize {adj} {med_term1} techniques for comprehensive {med_term2} management",
            f"Modern {med_term1} methodologies integrate {adj} technology with traditional {med_term2} practices",
            f"Specialized {med_term1} centers provide {adj} {med_term2} services for complex medical conditions"
        ]
        
        return random.choice(variations)
    
    def generate_compound_words(self):
        """Generate sentences with compound words and technical compounds."""
        compounds = []
        
        # Generate technical compounds
        for prefix in self.prefixes[:5]:
            for base in self.technical_terms[:10]:
                compounds.append(f"{prefix}{base}")
        
        # Generate domain compounds
        bases = self.technical_terms + self.scientific_terms + self.business_terms
        for i in range(20):
            word1 = random.choice(bases)[:6]  # Truncate for readability
            word2 = random.choice(bases)[:8]
            compounds.append(f"{word1}{word2}")
        
        # Create sentences with compounds
        sentences = []
        for compound in compounds[:15]:
            adj = random.choice(self.technical_adjectives)
            term = random.choice(self.scientific_terms)
            sentences.append(f"The {adj} {compound} system facilitates enhanced {term} capabilities")
        
        return sentences
    
    def generate_diverse_dataset(self, filename: str, target_size_gb: float = 4.0):
        """Generate highly diverse dataset with varied vocabulary."""
        logger.info(f"Generating diverse dataset: {filename} (target: {target_size_gb:.1f} GB)")
        
        target_bytes = int(target_size_gb * 1024 * 1024 * 1024)
        current_bytes = 0
        
        sentence_generators = [
            self.generate_technical_sentence,
            self.generate_scientific_sentence,
            self.generate_business_sentence,
            self.generate_academic_sentence,
            self.generate_medical_sentence
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Start with compound words for variety
            compound_sentences = self.generate_compound_words()
            for sentence in compound_sentences:
                f.write(sentence + '\n')
                current_bytes += len(sentence.encode('utf-8')) + 1
            
            # Generate diverse sentences
            cycle = 0
            while current_bytes < target_bytes:
                # Rotate through different generators
                generator = sentence_generators[cycle % len(sentence_generators)]
                
                try:
                    sentence = generator()
                    f.write(sentence + '\n')
                    current_bytes += len(sentence.encode('utf-8')) + 1
                    
                    cycle += 1
                    
                    # Add some random technical jargon sentences
                    if cycle % 50 == 0:
                        # Create very technical sentences with lots of compound terms
                        tech_sentence = self._generate_super_technical_sentence()
                        f.write(tech_sentence + '\n')
                        current_bytes += len(tech_sentence.encode('utf-8')) + 1
                        
                    if cycle % 1000 == 0:
                        logger.info(f"Generated {current_bytes / (1024*1024):.1f} MB so far...")
                        
                except Exception as e:
                    logger.warning(f"Error generating sentence: {e}")
                    continue
        
        final_size_mb = current_bytes / (1024 * 1024)
        logger.info(f"High-quality dataset created: {filename} ({final_size_mb:.1f} MB)")
        return filename
    
    def _generate_super_technical_sentence(self):
        """Generate extremely technical compound-heavy sentences."""
        prefixes = random.sample(self.prefixes, 3)
        bases = random.sample(self.technical_terms + self.scientific_terms, 5)
        suffixes = random.sample(self.suffixes, 2)
        
        # Create complex compound words
        compound1 = f"{prefixes[0]}{bases[0]}{suffixes[0]}"
        compound2 = f"{prefixes[1]}{bases[1]}"
        compound3 = f"{bases[2]}{bases[3]}{suffixes[1]}"
        compound4 = f"{prefixes[2]}{bases[4]}"
        
        templates = [
            f"Heterogeneous {compound1} architectures enable {compound2} integration with {compound3} systems",
            f"Multi-dimensional {compound1} frameworks facilitate {compound2} optimization through {compound3} methodologies",
            f"Cross-platform {compound1} solutions leverage {compound2} capabilities for {compound3} enhancement",
            f"Enterprise-grade {compound1} implementations require {compound2} coordination with {compound3} infrastructure",
            f"Next-generation {compound1} platforms integrate {compound2} functionality through {compound3} interfaces"
        ]
        
        return random.choice(templates)

def main():
    generator = HighQualityDataGenerator()
    
    # Generate 4GB of highly diverse data
    dataset_file = generator.generate_diverse_dataset(
        filename="high_quality_training_data.txt",
        target_size_gb=4.0
    )
    
    print(f"High-quality diverse dataset created: {dataset_file}")
    
    # Sample some sentences to verify diversity
    print("\nSample sentences:")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10:
                print(f"  {line.strip()}")
            else:
                break

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()